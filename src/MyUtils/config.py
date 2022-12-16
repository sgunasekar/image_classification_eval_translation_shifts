import os

import numpy as np
import torch

# import models

default_cfg={
    "CIFAR10": {
        "dataset": "CIFAR10",
        "im_dim": 32,
        "num_classes":10,
        "in_channels":3,
        "root": os.path.join("data","CIFAR10"),
        "mean": np.array([0.4914, 0.4822, 0.4465]),
        "std": np.array([0.2470, 0.2435, 0.2616]),
    },
    "CIFAR100": {
        "dataset": "CIFAR100",
        "im_dim": 32,
        "num_classes": 100,
        "in_channels": 3,
        "root": os.path.join("data","CIFAR100"),
        "mean": np.array([0.5071, 0.4866, 0.4409]),
        "std": np.array([0.2673, 0.2564, 0.2762]),
    },
    "TINYIMAGENET": {
        "dataset": "TINYIMAGENET",
        "im_dim": 64,
        "num_classes":200,
        "in_channels":3,
        "root": os.path.join("data","TINYIMAGENET"),
        "mean": np.array([0.4802, 0.4481, 0.3975]),
        "std": np.array([0.2770, 0.2691, 0.2821]),        
    },
    "IMAGENET": {
        "dataset": "IMAGENET",
        "im_dim": 224,
        "num_classes": 1000,
        "in_channels":3,
        "root": os.path.join("data","IMAGENET"),
        "mean": np.array([0.485, 0.456, 0.406]),
        "std": np.array([0.229, 0.224, 0.225]),
    },
    "data": {
        "basic_augmentation": False,
        "advanced_augmentation":False,
        "padded":False,
        "centercrop": False,
        "translation_augmentation": 4,
        "reprob": 0.25
    },
    "training": {
        "resume": False,
        "checkpoint_folder": os.path.join("save","CIFAR10"),
        "checkpoint_file": None,
        "opt": 'sgd',
        "epochs": 200,
        "print_freq": 5000,
        "eval_epoch_freq": 1,
        "num_workers": 8,
        "batch_size": 128,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "save_dir": os.path.join("save")
    },
    "scheduler": {}
}

def parser_add_arguments(parser):
    parser.add_argument('--base-suffix', default="_", help="base file suffix (default:'_')")
    parser.add_argument('--model', default="resnet18", help="model name (default: resnet18)")
    parser.add_argument('--dataset', default='CIFAR10', help="dataset name (default: CIFAR10)")

    parser.add_argument('--seed', default=-1, type=int, help='random seed (default -1)')
    parser.add_argument('--resume', default=False, action='store_true',  help='resume from checkpoint (default: False)')
    parser.add_argument('--checkpoint_folder', default=default_cfg['training']['checkpoint_folder'], help='checkpoint folder name to resume from or to validate on in validate_only.py (ignored by main.py if resume is False).')
    parser.add_argument('--checkpoint_file', default=default_cfg['training']['checkpoint_file'], help='checkpoint filename within checkpoint folderto resume from or to validate on in validate_only.py (ignored by main.py if resume is False).')
    parser.add_argument('--save_init', default=False, action='store_true', help='flag to save initialization of model parameters (default: False)')
    parser.add_argument('--disable-tensorboard', default=False, action='store_true', help='flag to skip logging into tensorboard')

    ## Data args    
    parser.add_argument('--padded', default=False, action='store_true', help="flag to train on padded image with im_dim/4 padding on each side (default: False)")

    ## Data augmentation args
    parser.add_argument('--basic-augmentation', '--ba', action='store_true',  dest='basic_augmentation', help='flag to use basic augmentation (default: True)')
    parser.add_argument('--no-basic-augmentation', '--no-ba', default=True, action='store_false', dest='basic_augmentation', help='flag to not use basic augmentation')
    parser.set_defaults(basic_augmentation=True)
    parser.add_argument('--translation_augmentation', default=4, type=int,  help='padding to use with random crop if data augmentation is true (default: 4)')
    parser.add_argument('--advanced-augmentation', '--auto-autmentation', '--auto-aug', '--aa', dest='advanced_augmentation', default=False, action='store_true',  help='flag to use auto augmentation (default: False)')
    parser.add_argument('--standard-augmentation', '--std-aug', '--sa', default=False, action='store_true', dest='standard_augmentation', help='flag to use standard randomresizecrop augmentation used for testing on IMAGENET only. If set to true other augmentation flags like  --ba or --aa are ignored. (default: False)')
    parser.add_argument('--reprob', type=float, default=0.25,  help='Random erase prob. Note: Used only when advanced_augmentation is True, else ignored. (default: 0.25)')

    ## Mixup params used by timm.data.Mixup
    parser.add_argument('--use-mixup', default=False, action='store_true',  help='flag to use mixup/cutmix  (default: False)')
    parser.add_argument('--use-label-smoothing', default=False, action='store_true',  help='flag to use label smoothing only if use_mixup=False (ignored if use_mixup=True) (default: False)')
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both enabled (default: 1.0)')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled (default: 0.5)')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem".  (default: "batch")')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing to be used in Mixup/CutMix (default: 0.1)')

    ## Model args
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--batchnorm', '--bn', default=False, action='store_true', dest='batchnorm', help='flag to use batch normalization in convnets. Specify at most one of --bn, --gn, --ln. Note that non-convnet models always use layernorm. (default: False)')
    group2.add_argument('--groupnorm','--gn', default=False, action='store_true', dest='groupnorm', help='flag to use group normalization + weight standardization in convnets. Specify at most one of --bn, --gn, --ln. Note that non-convnet models always use layernorm. (default: False)')
    group2.add_argument('--layernorm','--ln', default=False, action='store_true', dest='layernorm', help='flag to use batch normalization in convnets. Specify at most one of --bn, --gn, --ln. Note that non-convnet models always use layernorm. (default: False)')
    parser.add_argument('--patchify', default=False, action='store_true', help='flag to use patchify stem in BiT ResNet models.')
    parser.add_argument('--dropout', '--do', type=float, help='dropout rate if relevant for model (See defaults in aplicable models.default_<model>_config)')
    parser.add_argument('--drop-path', '--so', type=float, help='stochastic depth rate if relevant for model (See defaults in aplicable models.default_<model>_config)')
    parser.add_argument('--resize', action='store_true', help='resize images to 224x224 (default:True)')
    parser.add_argument('--no-resize', action='store_false', dest='resize', help='do not resize images to 224x224')
    parser.set_defaults(resize=True)

    ## Training args
    parser.add_argument('-b', '--batch-size', default=default_cfg['training']['batch_size'], type=int, help='mini-batch size (default: %d)' %default_cfg['training']['batch_size'])
    parser.add_argument('--epochs', default=default_cfg['training']['epochs'], type=int, help='number of total epochs to run (default: %d)' %default_cfg['training']['epochs'])
    parser.add_argument('-j', '--workers', default=default_cfg['training']['num_workers'], type=int, help='number of data loading workers (default: %d)' %default_cfg['training']['num_workers'])
    parser.add_argument('--print-freq', '--print-frequency', default=default_cfg['training']['print_freq'], type=int, help='frequency of printing in training (default: %d)' %default_cfg['training']['print_freq'])
    parser.add_argument('--eval_epoch_freq', default=default_cfg['training']['eval_epoch_freq'], type=int, help='frequency of checking and saving best epoch during in training -- used only for IMAGENET (default: %d), other datasets are evaluated at every epoch' %default_cfg['training']['eval_epoch_freq'])


    ## Optimizer params used by timm.optim.create_optimizer
    parser.add_argument('--opt', '--optimizer', default=default_cfg['training']['opt'], type=str, help='optimizer name "sgd" or "Adamw" implemented currently (default: %s)' %default_cfg['training']['opt'])
    parser.add_argument('--lr', '--learning-rate', default=default_cfg['training']['lr'], type=float, help='initial learning rate (default: %0.2f)' %default_cfg['training']['lr'])
    parser.add_argument('--momentum', default=default_cfg['training']['momentum'], type=float, help='momentum parameter (default: %0.2f)' %default_cfg['training']['momentum'])
    parser.add_argument('--weight-decay', '--wd', default=default_cfg['training']['weight_decay'], type=float, help='weight decay (default: %0.4f)' %default_cfg['training']['weight_decay'])
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--clip-grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')

    ## Scheduler params used by timm.scheduler.create_scheduler
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler (default: "cosine")')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='lower lr bound for cyclic schedulers that hit 0 (default: 1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=20, help='epochs to warmup LR, if scheduler supports (default: 20 for non-imagenet, for imagenet, warmup is fixed to 5 epoch! )')
    parser.add_argument('--cooldown-epochs', type=int, default=0, help='epochs to cooldown LR at min_lr, after cyclic schedule ends (default: 0)')
    parser.add_argument('--decay-epochs', default = 100000, type=float, help='epoch interval to decay LR for step-like schedules (default: args.epochs)')
    parser.add_argument('--decay-rate', '--dr', default=0, type=float, help='LR decay rate for step-like schedules (default: 0.1)')

    ## Distributed training parameters
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use for training and testing')
    parser.add_argument('--use-amp', action='store_true', help='flag for using automatic mixed precision in training (default:False)')

    parser.add_argument('--dist-eval', action='store_true', help='flag for using distributed computation for validation (default: True)')
    parser.add_argument('--no-dist-eval', action='store_false', dest='dist_eval')
    parser.set_defaults(dist_eval=True)

    parser.add_argument('--repeated-aug', action='store_true', help='flag for using repeated augmentation (default:True in distributed)')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--dist_backend', default='nccl', help='backend to use for distributed training (default: nccl)')
    #parser.add_argument('--local_rank', default=0, type=int, help='local gpu id (default = 1)')

    return parser
