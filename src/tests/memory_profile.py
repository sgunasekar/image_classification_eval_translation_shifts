import argparse
import logging
import os
import sys

import torch
import torch.cuda
import torch.nn as nn
from timm.optim import create_optimizer
from torch.cuda.amp import GradScaler
from torch.profiler import (ProfilerActivity, profile, record_function,
                            tensorboard_trace_handler)

logger = logging.getLogger()
log_file = os.path.join('log' ,f"memory_profiling_logs.log")
# Setup logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s %(filename)s:%(lineno)d] >> %(message)s',
    datefmt='%y-%m-%d:%H:%M', 
    handlers= [
        logging.FileHandler(log_file,'w'),
    ],
    level=logging.INFO
)

def single_update(X, y, model, optimizer, loss_scaler=None, criterion=nn.CrossEntropyLoss(), device='cuda', SW=None):

    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    model = model.to(device)

    optimizer.zero_grad()
    with record_function("forward pass"):
        with torch.cuda.amp.autocast(enabled=(loss_scaler is not None)):

            output = model(X)
            loss = criterion(output, y)
    with record_function("backward pass"):
        if not loss_scaler:
            loss.backward()
            optimizer.step()
        else:
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            
def profile_single_update(batch_size, model, optimizer, loss_scaler, config_str="test"):
    

    
    wait = 1
    warmup = 1
    active = 3
    repeat = 1
    num_steps = repeat*(wait+warmup+active)
    with profile(
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, 
        active=active, repeat=repeat),
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(os.path.join('log', config_str))) as prof:
        
        for i in range(num_steps):

            with record_function("single_update"):
                X = torch.randn(batch_size, 3, 224, 224)
                y = torch.randint(0,1000, (batch_size,))
            
                torch.cuda.reset_max_memory_allocated()
                single_update(X, y, model, optimizer, loss_scaler=loss_scaler)
                peak = torch.cuda.max_memory_allocated()                
                if i == wait+warmup+active//2: 
                    logger.info(f"Peak memory of {config_str} = {peak/(10**9)}")

            prof.step()


class _opt_args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

opt_args = dict(
    sgd = _opt_args(opt='momentum', lr=0.1, weight_decay=0.0001, momentum=0.9),
    sgd_no_momentum = _opt_args(opt='momentum', lr=0.1, weight_decay=0.0001, momentum = 0.0),
    adamw = _opt_args(opt='adamw', lr=0.001, weight_decay=0.1, momentum=0.0),
)


# The following is a sample code to work with above code (replace model/optimizer definition with custom ones)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from MyUtils.model_cfg import get_model_cfg

if __name__ == '__main__':

    for model_name in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge']:

        for opt_key in ['sgd','adamw','sgd_no_momentum']:

            for loss_scaler in [None,GradScaler()]:

                # define model
                model_cfg, model_name = get_model_cfg(model_name)
                assert model_cfg is not None, "Could not process model_cfg"
                Net = model_cfg.pop('net_class')
                model_cfg['im_dim'] = 224
                model_cfg['num_classes'] = 1000
                model_cfg['in_channels'] = 3
                print("Model cfg:", model_cfg)
                model = Net(model_cfg).vit_model # REPLACE with custom model

                # define optimizer
                optimizer = create_optimizer(args=opt_args[opt_key], model=model) # REPLACE with custom optimizer

                print("Setup...")
                print(f"Loss scaler={loss_scaler}")
                number_trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
                print(f"Model: {model_name}, no. of trainable parameters={number_trainable_parameters}")
                print(f"Optimizer: {opt_key}")
                
                for batch_size in [1,4,16]:
                    
                    # CHK: for vit models from timm, can use set_grad_checkpointing(enable=True) for checkpointing
                    for checkpoint in [True, False]:

                        model.set_grad_checkpointing(enable=checkpoint)

                        profile_single_update(batch_size, model, optimizer, loss_scaler, config_str=f"opt_{opt_key}_model_{model_name}_b_{batch_size}_checkpoint_{checkpoint}_amp_{loss_scaler is not None}")