import argparse
import logging
import os
import time
from datetime import datetime
from shutil import copyfile, copytree

from MyUtils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

parser = argparse.ArgumentParser(description='translation generalization')

parser_add_arguments(parser)

#########################################################################################
def main(args):

    print('====================START==============')    
    print("Number of gpus: ", torch.cuda.device_count())

    ### DDP modification: init_distributed_model will set args.distributed=True if running in distributed mode;
    ### sets torch.cuda.use_device() and makes all print only come from main process henceforth
    distributed.init_distributed_mode(args)

    device = torch.device(args.device)
    using_gpu = device.type =='cuda'

    if using_gpu: torch.backends.cudnn.benchmark = True #https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not

    if args.seed>=0:
        seed = args.seed + distributed.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if using_gpu: torch.cuda.manual_seed(seed)

    ## Process model configuration
    model_cfg, model_name = get_model_cfg(args)
    assert model_cfg is not None, "Could not process model_cfg"
    Net = model_cfg.pop('net_class')
    suffix = model_name+args.base_suffix

    ## Process data configuration
    dataset = args.dataset.upper()
    if dataset!='IMAGENET':
        data_cfg, suffix = get_data_cfg(suffix, args)
    else:
        data_cfg, suffix = get_imagenet_data_cfg(suffix, args)
        
    model_cfg['im_dim'] = data_cfg['im_dim']
    model_cfg['num_classes'] = data_cfg['num_classes']
    model_cfg['in_channels'] = data_cfg['in_channels']

    ## Setup directories and files
    data_cfg['root'] = os.environ.get('DATA_DIR', default_cfg[dataset]['root'])
    suffix = suffix+"_se%d" %args.seed
    
    if args.checkpoint_file is None:
        print("Checkpoint folder:", args.checkpoint_folder)
        lastepoch_file = None
        bestepoch_file = None
        checkpoint_files = os.listdir(args.checkpoint_folder)
        print("Files in checkpoint folder:", checkpoint_files)
        for filename in checkpoint_files:
            if filename.startswith('lastepoch'):
                lastepoch_file = os.path.join(args.checkpoint_folder, filename)
            if filename.startswith('bestepoch'):
                bestepoch_file = os.path.join(args.checkpoint_folder,filename)
        
        assert lastepoch_file is not None, "did not find lastepoch_file"
        # Hack
        if '_BN' in lastepoch_file:
            model_cfg['batchnorm']=True
        elif '_GN' in lastepoch_file:
            model_cfg['groupnorm']=True
    else:
        lastepoch_file = os.path.join(args.checkpoint_folder, args.checkpoint_file)
        bestepoch_file = None
    
    ## Print setup configuration
    print("-------Configuration------")
    print("Using gpu:", using_gpu)
    print("Using distributed:", args.distributed, distributed.get_world_size(), distributed.get_rank())
    print("Data dir:", data_cfg['root'])
    print("Lastepoch file:", lastepoch_file)
    print("Bestepoch file:", bestepoch_file)

    print("Model: ", model_cfg)
    print("Data: ", data_cfg)
    print("Args: ", args)

    ##########################################################################

    print("---Setup...---")
    m = torch.cuda.memory_allocated()
    data = Dataset(data_cfg = data_cfg, val_only = True)
    model = Net(model_cfg)
    model.to(device)
    with torch.no_grad():
        kwargs = {'force':True} if args.distributed else {}
        print("Model in device:", next(model.parameters()).device, **kwargs)
    print("\t Memory: model %d --> %d" %(m,torch.cuda.memory_allocated()))

    ### DDP modification: Wrap model in DDP
    ### DDP modification: Create distributed data loaders
    ### DDP modification: Scale learning rates
    ### Note: DDP by default averages the gradients from multiple processes. In DDP the effective batch_size=num_tasks*args.batch_size
    if args.distributed:
        num_tasks = distributed.get_world_size()
        global_rank = distributed.get_rank()

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

        if args.dist_eval:
            if len(data.testset) % num_tasks != 0:
                logging.warning("Enabling distributed evaluation with an eval dataset not divisible by process number. This will slightly alter validation results as extra duplicate entries are added to achieve equal num of samples per-process.")
            sampler_test = torch.utils.data.distributed.DistributedSampler(data.testset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(data.testset)

    else:
        sampler_test = torch.utils.data.SequentialSampler(data.testset)
        model_without_ddp = model


    testloader = torch.utils.data.DataLoader(
        data.testset, sampler=sampler_test,
        batch_size=int(1.5*args.batch_size),
        num_workers=args.workers,
        pin_memory=using_gpu,
        drop_last=False
    )

    print(model)
    print("No. of parameters: ", count_all_parameters(model), " ",  count_trainable_parameters(model))

    criterion = get_criterion(args)
    criterion.to(device) 
    
    test_stats = dict(loss=[], prec1=[])
    
    # Update model
    trained_model = torch.load(lastepoch_file, map_location='cpu')
    print(trained_model.keys())
    if isinstance(trained_model, dict):
        model_without_ddp.load_state_dict(trained_model['model_state_dict'])
        best_testprec1 = trained_model.get('best_testprec1', 0)
    else:
        model_without_ddp.load_state_dict(trained_model)
        best_testprec1 = 0
    
    # Validation
    with torch.no_grad():
        epoch_test_stats = validate(testloader = testloader, model=model, criterion = nn.CrossEntropyLoss(), device=device)

        if isinstance(trained_model, dict) and 'test_stats' in trained_model:
            for key in test_stats: 
                trained_model['test_stats'][key].append(epoch_test_stats[key])

        ### DDP modification: save only in main process
        ## Save checkpoint
        if (epoch_test_stats['prec1'] > best_testprec1):
            best_testprec1 = epoch_test_stats['prec1']            
            best_testprec1_epoch = args.epochs-1
            trained_model['best_testprec1'] = best_testprec1
            trained_model['best_testprec1_epoch'] = best_testprec1_epoch
            
            if distributed.is_main_process() and bestepoch_file is not None and isinstance(trained_model, dict):
                torch.save({
                    'epoch': args.epochs-1,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': trained_model['optimizer_state_dict'],
                    'scheduler_state_dict': trained_model['scheduler_state_dict'],
                    'best_testprec1': best_testprec1,
                    }, bestepoch_file)
            
        if distributed.is_main_process():
            torch.save(trained_model, lastepoch_file)

                

    print('End of validation: Best val prec1: %0.4g (epoch %d), last val prec1 %0.4g' %(best_testprec1, best_testprec1_epoch, epoch_test_stats['prec1']))
    print("================DONE===========",flush=True)

    distributed.cleanup_distributed_mode()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
