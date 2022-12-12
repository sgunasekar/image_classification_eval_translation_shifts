# README

The repository is a companion to the paper [Generalization to translation shifts: a study in architectures and augmentations](https://arxiv.org/abs/2207.02349). More generally, the code supports multi-GPU distributed training of standard image classification models: VGG, ResNe(X)ts, ViTs, (Res)MLP-mixer. Much of the code is derived from the excellent [timm](https://github.com/rwightman/pytorch-image-models) library, with additional adaptations from other open source respositories, notably [DeiT](https://github.com/facebookresearch/deit) and [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).


Run `pip install requirements.tex` to install necessary packages.

Check MyUtils/config.py for full list options that can be passed as command line arguments and their default arguments. Below is a sample use case for training.

### Image modifications
The image preprocessing options that can be passed through commandline are 
- `--padded` adds a mean-padded-canvas which was used for evaluating translation shifts in a controlled manner [here](https://arxiv.org/abs/2207.02349). Size of padding is set to 1/4th of image size and is set in MyUtils/datautils.py.
- `--resize` resizes the input images to $224\times 224$ which is the standard for ImageNet.

### Training
A sample command to train models in distributed mode:
`python -m torch.distributed.launch --nproc_per_node=[number of GPUS] --use_env main.py --padded --dataset [dataset_name] --model [model_name] -b [batch_size] --opt [sgd or adamw] --lr [learning_rate] --wd [weight_decay] --epochs [epochs]`

To use in non-distributed mode, simply use `python main.py [--arg1 --arg2 --arg3]`

- Supported dataset names are : CIFAR10, CIFAR100, TINYIMAGENET, and IMAGENET. Use commanline argument `--dataset [dataset name]` to specify the dataset
- The following are a subset of the model names that are supported using commanline argument `--model [model_name]`. For each model class, models/[model_class].py lists all valid model names at the top.
  - vgg11, vgg13, vgg16, vgg19
  - resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2, preact_resnet18, preact_resnet50
  - cait_xxs24, cait_xxs36, cait_xs24, cait_s24, cait_s36, cait_m36, cait_m48, vit_tiny, vit_small, vit_base, deit_tiny, deit_small, deit_base
  - resmlp_12_224, resmlp_36_224, resmlp_big_24_224, mixer_s32_224, mixer_s16_224, mixer_b32_224, mixer_b16_224, mixer_l32_224, mixer_l16_224
- Use `--batchnorm` or `--groupnorm` to use respective norms in convolutional networks.
- Use the following commanline options for the different data augmentation pipelines
  - No data augmentation (None): `--no-basic-augmentation`
  - Basic augmentation (BA): `--basic-augmentation`
  - Advanced augmentation (AA): `--basic-augmentation --advanced-augmentation --use-mixup`
  - AA (no tr): `--no-basic-augmentation --advanced-augmentation --use-mixup`
- Check MyUtils/config.py for full list options that can be passed as command line arguments and their defaults
- Additional setup: Edit the paths for `default_cfg['[dataset_name]']['root']` and `default_cfg['training']['save']` options in the default_cgfs dictionary to point to the dataset root director and save directory for saving results.


### Evaluation
To evaluate on translation generalization grid, use  translation_generalization.py with the command line arguments specified in the file. Easiest way to provide trained model checkpoint file is to change the default `model_filename` declaration in the top of translation_generalization.py.

### TODO

- [ ] add support for ImageNet21k
