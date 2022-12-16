import argparse
import logging
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from MyUtils import *

parser = argparse.ArgumentParser()

parser_add_arguments(parser)

if __name__ == '__main__':
    
    print("Pass the same commandline arguments as main.py")
    args = parser.parse_args()

    print("Number of gpus: ", torch.cuda.device_count())
    ### DDP modification: init_distributed_model will set args.distributed=True if running in distributed mode;
    ### sets torch.cuda.use_device() and makes all print only come from main process henceforth
    distributed.init_distributed_mode(args)

    device = torch.device(args.device)
    using_gpu = device.type =='cuda'

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
    # Change the scheduler to use per step update
    if dataset=='IMAGENET':
        args.sched_on_updates = True
        args.warmup_epochs = 5

    ## Process special training arguments
    train_kwargs = {}
    train_kwargs['num_classes'] = model_cfg['num_classes']
    train_kwargs, suffix = get_train_kwargs(train_kwargs, args, suffix)

    ## Setup directories and files
    data_cfg['root'] = os.environ.get('DATA_DIR', default_cfg[dataset]['root'])
    output_dir = os.path.join(os.environ.get('OUTPUT_DIR', default_cfg['training']['save_dir']))
    save_dir = output_dir
    
    suffix = suffix+"_se%d" %args.seed

    lastepoch_file = os.path.join(output_dir,'lastepoch.pt')
    bestepoch_file = os.path.join(save_dir,'bestepoch.pt')

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