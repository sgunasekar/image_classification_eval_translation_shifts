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

    ## Process special training arguments
    train_kwargs = {}
    train_kwargs['num_classes'] = model_cfg['num_classes']
    train_kwargs, suffix = get_train_kwargs(train_kwargs, args, suffix)

    ## Setup directories and files
    data_cfg['root'] = os.environ.get('DATA_DIR', default_cfg[dataset]['root'])
    output_dir = os.path.join(os.environ.get('OUTPUT_DIR', default_cfg['training']['save_dir']))
    save_dir = output_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    suffix = suffix+"_se%d" %args.seed

    lastepoch_file = os.path.join(output_dir,'lastepoch_%s.pt' %(suffix))
    bestepoch_file = os.path.join(save_dir,'bestepoch_%s.pt' %(suffix))
    if args.save_init:
        initalization_file = os.path.join(output_dir,"initialization_%s.pt" %(suffix))
    ### DDP: log only in main process
    if not(args.disable_tensorboard) and distributed.is_main_process():
        SW = SummaryWriter(os.path.join(output_dir,"tensorboard_%s" %datetime.now().strftime("%Y%m%d-%H%M%S")))
        train_kwargs['SW']=SW

    ## Print setup configuration
    print("-------Configuration------")
    print("Using gpu:", using_gpu)
    print("Using distributed:", args.distributed, distributed.get_world_size(), distributed.get_rank())
    print("Data dir:", data_cfg['root'])
    print("Save dir:", save_dir)
    print("Output dir:", output_dir)
    print("Lastepoch file:", lastepoch_file)
    print("Bestepoch file:", bestepoch_file)

    print("Train parameters: batchsize=%d, epochs=%d, num_workers=%d,  (pin_memory=%d,lr=%0.2g, momentum=%0.2g, weight_decay=%0.2g)" %(args.batch_size,args.epochs,args.workers,int(using_gpu),args.lr,args.momentum, args.weight_decay))
    print("Model: ", model_cfg)
    print("Data: ", data_cfg)
    print("Train kwargs: ",train_kwargs)
    print("Args: ", args)

    ##########################################################################

    print("---Setup...---")
    m = torch.cuda.memory_allocated()
    data = Dataset(data_cfg = data_cfg)
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

        linear_scaled_lr = args.lr * num_tasks * args.batch_size / 512.0
        args.lr = linear_scaled_lr

        if args.repeated_aug:
            sampler_train = distributed.RASampler(data.trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.distributed.DistributedSampler(data.trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        if args.dist_eval:
            if len(data.testset) % num_tasks != 0:
                logging.warning("Enabling distributed evaluation with an eval dataset not divisible by process number. This will slightly alter validation results as extra duplicate entries are added to achieve equal num of samples per-process.")
            sampler_test = torch.utils.data.distributed.DistributedSampler(data.testset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_test = torch.utils.data.SequentialSampler(data.testset)

    else:
        sampler_train = torch.utils.data.RandomSampler(data.trainset)
        sampler_test = torch.utils.data.SequentialSampler(data.testset)
        model_without_ddp = model

    print("\t Memory: model after ddp %d" %(torch.cuda.memory_allocated()))
    trainloader = torch.utils.data.DataLoader(
        data.trainset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=using_gpu,
        drop_last=True,
    )

    testloader = torch.utils.data.DataLoader(
        data.testset, sampler=sampler_test,
        batch_size=int(1.5*args.batch_size),
        num_workers=args.workers,
        pin_memory=using_gpu,
        drop_last=False
    )

    print("\t Memory: data loader %d" %torch.cuda.memory_allocated())

    print(model)
    print("No. of parameters: ", count_all_parameters(model), " ",  count_trainable_parameters(model))

    criterion = get_criterion(args)
    criterion.to(device) 
    
    optimizer = create_optimizer(args = args, model = model_without_ddp) ## Note: For distributed, this is ok as the synchronization happens at loss.backward() and not at optimizer.step()
    scheduler, _ = create_scheduler(args = args, optimizer = optimizer)

    print("Criterion, optimizer, scheduler")
    print(criterion, optimizer, scheduler)
    ### DDP modification: save only in main process
    if args.save_init and distributed.is_main_process():
        torch.save({
                    'model_cfg':model_cfg,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, initalization_file)


    train_stats = dict(loss=[], prec1=[], lr=[], grad_clip_count=[], param_norm=[], grad_norm=[])
    test_stats = dict(loss=[], prec1=[])
    best_testprec1 = 0 
    best_testprec1_epoch = -1

    start_epoch = 0

    if args.resume:
        start_epoch, train_stats, test_stats, best_testprec1, best_testprec1_epoch = load_checkpoint(args,model_without_ddp, optimizer, scheduler, train_kwargs['loss_scaler'])


    print("---Begin training: memory %d...---" %torch.cuda.memory_allocated())
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):

        ### DDP modifications: set trainloader epoch
        ### DDP modification: within train_epoch and validate, make sure to synchronize aggregates/Average meters
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)

        m = torch.cuda.memory_allocated()
        epoch_train_stats = train_epoch(trainloader = trainloader, model = model, optimizer = optimizer, epoch=epoch, criterion = criterion, device = device, **train_kwargs)
        if epoch<5: print("\t Memory: training %d --> %d" %(m,torch.cuda.memory_allocated()))

        ## Note: For last ten epochs, compute full training loss and prec1
        ## In other epochs, we get running average whenever mixup is off.
        if epoch>=args.epochs-10 and dataset!='IMAGENET':
            epoch_full_train_stats = validate(trainloader, model, criterion = nn.CrossEntropyLoss(), device=device)
            for key in epoch_full_train_stats: epoch_train_stats[key] = epoch_full_train_stats[key]

        ### DDP modification: log only in main process
        if not(args.disable_tensorboard) and distributed.is_main_process():
            SW.add_scalar('Loss/Train', epoch_train_stats['loss'], epoch)
            SW.add_scalar('Accuracy/Train', epoch_train_stats['prec1'], epoch)
            # SW.add_scalar('Epochtime/Train', epoch_train_stats['epochtime'], epoch)
            SW.add_scalar('Hyperparameter/lr', epoch_train_stats['lr'], epoch)
            SW.add_scalar('Misc/grad_norm', epoch_train_stats['grad_norm'], epoch)
            SW.add_scalar('Misc/param_norm', epoch_train_stats['param_norm'], epoch)
            SW.add_scalar('Misc/grad_clip_count', epoch_train_stats['grad_clip_count'], epoch)
            SW.add_scalar('Misc/cuda_alloc_memory', epoch_train_stats['cuda_alloc_memory'], epoch)

        for key in train_stats: train_stats[key].append(epoch_train_stats[key])

        # Validation    
        m = torch.cuda.memory_allocated()
        if dataset!='IMAGENET' or not(epoch%args.eval_epoc_freq):
            with torch.no_grad():
                epoch_test_stats = validate(testloader = testloader, model=model, criterion = nn.CrossEntropyLoss(), device=device)
            if epoch<5: print("\t Memory: validation %d --> %d" %(m,torch.cuda.memory_allocated()))

            ### DDP modification: log only in main process
            if not(args.disable_tensorboard) and distributed.is_main_process():
                SW.add_scalar('Loss/Test', epoch_test_stats['loss'], epoch)
                SW.add_scalar('Accuracy/Test', epoch_test_stats['prec1'], epoch)

            for key in test_stats: test_stats[key].append(epoch_test_stats[key])

            ### DDP modification: save only in main process
            ## save checkpoint
            if (epoch_test_stats['prec1'] > best_testprec1):
                best_testprec1 = epoch_test_stats['prec1']
                best_testprec1_epoch = epoch
                if distributed.is_main_process():
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_without_ddp.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_testprec1': best_testprec1,
                        }, bestepoch_file)

        if np.isnan(epoch_train_stats['loss']):
            print('EXITING DUE TO NAN. lr=%0.4g' %get_lr(optimizer))
            break

        scheduler.step(epoch)

    if distributed.is_main_process():
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_scaler_state_dict': train_kwargs['loss_scaler'].state_dict() if args.use_amp else None,
            'train_stats': train_stats,
            'test_stats': test_stats,
            'best_testprec1': best_testprec1,
            'best_testprec1_epoch': best_testprec1_epoch
            }, lastepoch_file)

    print('End of training: Best val prec1: %0.4g (epoch %d); Train prec1: %0.4g' %(best_testprec1, best_testprec1_epoch, np.mean(train_stats['prec1'][-5:])))
    total_time = time.time() - start_time
    print('Total training time:%0.3f hrs' %(total_time/3600.0))

    print("================DONE===========",flush=True)

    if distributed.is_main_process():
        if not(args.disable_tensorboard):
            SW.flush()
        if output_dir!= save_dir:
            if not os.path.exists(output_dir):
                print('Creating output directory: %s' %output_dir)
                os.makedirs(output_dir)
            print('Copying bestepoch from %s to %s' %(bestepoch_file,output_dir))
            copyfile(bestepoch_file,os.path.join(output_dir,"bestepoch_%s.pt" %(suffix)))
            # os.remove(bestepoch_file)

    distributed.cleanup_distributed_mode()

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        logging.warning("Cannot import SummaryWriter from tensorboard. Disabling tensorboard logging!")
        args.disable_tensorboard = True
    main(args)
