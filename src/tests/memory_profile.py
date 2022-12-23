import argparse
import logging
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from MyUtils.model_cfg import get_model_cfg
from torch.cuda.amp import GradScaler
from timm.optim import create_optimizer
from MyUtils.utils import count_trainable_parameters



def single_update(X, y, model, optimizer, loss_scaler=None, criterion=nn.CrossEntropyLoss(), device='cuda', SW=None):
    print(device)
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
            
def profile_single_update(batch_size, model, optimizer, loss_scaler):
    
    print("Setup...")
    print(f"Loss scaler={loss_scaler}, batch_size={batch_size},")

    print(f"Model: {model_name}, no. of trainable parameters={count_trainable_parameters(model)}")

    print(f"Optimizer: {optimizer}")
    
    wait = 0
    warmup = 0
    active = 1
    repeat = 1
    num_steps = repeat*(wait+warmup+active)
    with profile(
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler('log'),
        use_cuda=True) as prof:
        
        for i in range(num_steps):
            with record_function("single_update"):
                X = torch.randn(batch_size, 3, 224, 224)
                y = torch.randint(0,1000, (batch_size,))
                single_update(X, y, model, optimizer, loss_scaler=loss_scaler)
            prof.step()

        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=5))

class _opt_args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

opt_args = dict(
    sgd = _opt_args(opt='sgd', lr=0.1, weight_decay=0.0001, momentum=0.9),
    sgd_no_momentum = _opt_args(opt='sgd', lr=0.1, weight_decay=0.0001, momentum = 0.0),
    adamw = _opt_args(opt='adamw', lr=0.001, weight_decay=0.1),
)

if __name__ == '__main__':
    
    for loss_scaler in [None,GradScaler()]:

        for model_name in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            model_cfg, model_name = get_model_cfg(model_name)
            assert model_cfg is not None, "Could not process model_cfg"
            Net = model_cfg.pop('net_class')
            model_cfg['im_dim'] = 224
            model_cfg['num_classes'] = 1000
            model_cfg['in_channels'] = 3
            print("Model cfg:", model_cfg)
            model = Net(model_cfg)
    
    
            for opt_key in ['sgd','adamw','sgd_no_momentum']:
                optimizer = create_optimizer(args=opt_args[opt_key], model=model)
                
                for batch_size in [1,4,16]:
                    
                    profile_single_update(batch_size, model, optimizer, loss_scaler)

                    break
                break
            break
        break

