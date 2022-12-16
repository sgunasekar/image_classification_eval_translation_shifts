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
    
    model_cfg, model_name = get_model_cfg(args)
    assert model_cfg is not None, "Could not process model_cfg"
    Net = model_cfg.pop('net_class')
    
    dataset = args.dataset.upper()
    if dataset!='IMAGENET':
        data_cfg, suffix = get_data_cfg("", args)
    else:
        data_cfg, suffix = get_imagenet_data_cfg("", args)
        
    model_cfg['im_dim'] = data_cfg['im_dim']
    model_cfg['num_classes'] = data_cfg['num_classes']
    model_cfg['in_channels'] = data_cfg['in_channels']

    print("Model cfg:", model_cfg)
    model = Net(model_cfg)
    print(model)
