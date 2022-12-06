import os
import tarfile

import numpy as np
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from .autoaugment import rand_augment_transform
from .config import *

#### Dataset ####


class Dataset():

    def __init__(self, data_cfg, download=True, print_cfg= True, val_only=False):

        # Default values
        root = data_cfg['root']
        dataset = data_cfg['dataset']

        self.transform = data_cfg['transform']
        self.test_transform = data_cfg.get('test_transform',self.transform)

        self.mean = data_cfg['mean']
        self.std = data_cfg['std']

        if not os.path.exists(root):
            print("Data directory %s does not exist: creating the directory and downloading dataset %s" %(root,dataset))
            os.makedirs(root)
            download = True
            
        if print_cfg:
            print('Dataset: %s from %s' %(dataset,root))
            print('train_transform: ', self.transform)
            if not(isinstance(self.test_transform,list)):
                print('test_transform: ', self.test_transform)
            else:
                print('test_transform: ', self.test_transform[0])

        if dataset == 'MNIST':
            
            if not(val_only):
                self.trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=self.transform)
                self.classes = (['%d' %i for i in range(10)])
                self.D = self.trainset.__getitem__(1)[0].shape
            
            self.testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=self.test_transform)
            

        elif dataset == 'CIFAR10':
            
            if not(val_only):
                self.trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=self.transform)
                self.classes = np.array(['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
                self.D = self.trainset.__getitem__(1)[0].shape

            self.testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=self.test_transform)

        elif dataset == 'CIFAR100':
            
            if not(val_only):
                self.trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=self.transform)
                self.classes = np.array(['%d' %i for i in range(100)])
                self.D = self.trainset.__getitem__(1)[0].shape

            self.testset = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=self.test_transform)
            
        elif dataset == 'TINYIMAGENET':
            
            if not(val_only):
                self.trainset = torchvision.datasets.ImageFolder(root=os.path.join(root,"train"), transform=self.transform)
                self.classes = np.array(self.trainset.classes, dtype=object)
                self.D = self.trainset.__getitem__(1)[0].shape
        
            try:
                self.testset = torchvision.datasets.ImageFolder(root=os.path.join(root,"val"), transform=self.test_transform)
            except:
                print("Error: Val folder of tiny-imagenet needs to be preprocessed to have images stored in  format '/<val_folder>/<class_name>/[...]/<file_name>' that is accepted by torch.datasets.ImageFolder. Please check.")

        elif dataset == 'IMAGENET':
            
            if not(val_only):
                self.trainset = torchvision.datasets.ImageNet(root=root, split='train', transform=self.transform)
                self.classes = np.array(self.trainset.classes, dtype=object)
                self.D = self.trainset.__getitem__(1)[0].shape            
                        
            self.testset = torchvision.datasets.ImageNet(root=root, split='val', transform=self.test_transform)
  
def get_data_cfg(suffix, args):
    
    dataset = args.dataset.upper()
    data_cfg = default_cfg[dataset]
    

    im_dim = data_cfg['im_dim']
    mean = data_cfg['mean']
    std = data_cfg['std']
    aa_config_string = 'rand-m5-mstd0.5-inc1-tr0'#'rand-m9-n3-mstd0.5-inc1-tr0'

    data_transforms = [transforms.ToTensor(), transforms.Normalize(mean,std)]
    data_pad_transforms = []
    randcrop_padding = args.translation_augmentation

    #defaults
    data_cfg['transform'] = transforms.Compose(data_transforms)
    data_cfg['test_transform'] = transforms.Compose(data_transforms)
    fill = tuple([min(255, int(round(255 * x))) for x in mean])   
    
    if args.padded:
        suffix = suffix +"_padded"
        pad_size = int(0.25*im_dim)
        im_dim = im_dim+2*pad_size        
                     
        data_pad_transforms = [transforms.Pad(pad_size, fill=fill)]
        
        data_cfg['im_dim'] = im_dim
        data_cfg['test_transform'] = transforms.Compose(data_pad_transforms+data_transforms)
        data_cfg['transform'] = transforms.Compose(data_pad_transforms+data_transforms)


    if args.basic_augmentation or args.advanced_augmentation:
        if args.basic_augmentation:
            basic_data_augmentation_transforms = [transforms.RandomCrop(im_dim, padding=randcrop_padding, fill=fill), transforms.RandomHorizontalFlip()]
        else:
            basic_data_augmentation_transforms = [transforms.RandomHorizontalFlip()]
            aa_config_string = aa_config_string+'-sh0'

        if args.advanced_augmentation:
            data_cfg['aa_config_string'] = aa_config_string
            aa_params = dict(translate_const=int(im_dim * 0.45), img_mean=tuple([min(255, round(255 * channel_mean)) for channel_mean in mean]))
            rand_augmentation_transforms = [rand_augment_transform(aa_config_string, aa_params)]
            # [transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10)]
            re_transform = [transforms.RandomErasing(p=args.reprob, value='random')] if args.reprob > 0.0 else []
            transforms_list = rand_augmentation_transforms\
                + data_pad_transforms \
                + basic_data_augmentation_transforms\
                + data_transforms\
                + re_transform
            suffix = suffix +"_AA" if args.basic_augmentation else suffix +"_AAtr0"

        else:
            transforms_list = data_pad_transforms \
                + basic_data_augmentation_transforms\
                + data_transforms
            suffix = suffix +"_DA"

        data_cfg['transform'] = transforms.Compose(transforms_list)    

    return data_cfg, suffix
    
def get_imagenet_data_cfg(suffix, args):
    
    dataset = 'IMAGENET'
    data_cfg = default_cfg[dataset]

    im_dim = data_cfg['im_dim']
    mean = data_cfg['mean']
    std = data_cfg['std']
    aa_config_string = 'rand-m9-n3-mstd0.5-inc1-tr0'
    max_im_dim = im_dim+int(0.5*im_dim) # 336
    randcrop_padding = args.translation_augmentation
    rand_crop_dim = im_dim+2*randcrop_padding
 
    test_resize_transforms = [
        transforms.Resize(max_im_dim), 
        transforms.CenterCrop(im_dim),  # 224
        ]     
    train_resize_transforms = [
        transforms.Resize(max_im_dim), 
        transforms.CenterCrop(rand_crop_dim), # 224 + 4*2 = 232
        ]     
    data_transforms = [
        transforms.ToTensor(), 
        transforms.Normalize(mean,std)
    ]    
    
    #defaults
    data_cfg['transform'] = transforms.Compose(test_resize_transforms+data_transforms)
    data_cfg['test_transform'] = transforms.Compose(test_resize_transforms+data_transforms)
    fill = tuple([min(255, int(round(255 * x))) for x in mean])   
    

    if args.basic_augmentation or args.advanced_augmentation or args.standard_augmentation:
        if args.standard_augmentation:
            scale = tuple((0.08, 1.0))  # default imagenet scale range
            ratio = tuple((3./4., 4./3.))  # default imagenet ratio range
            basic_data_augmentation_transforms = [transforms.RandomResizedCrop(im_dim, scale=scale, ratio=ratio), transforms.RandomHorizontalFlip()]
            train_resize_transforms = []
        elif args.basic_augmentation:
            basic_data_augmentation_transforms = [transforms.RandomCrop(im_dim), transforms.RandomHorizontalFlip()]
        else:
            basic_data_augmentation_transforms = [transforms.CenterCrop(im_dim), transforms.RandomHorizontalFlip()]
            aa_config_string = aa_config_string+'-sh0'

        if args.advanced_augmentation:
            data_cfg['aa_config_string'] = aa_config_string
            aa_params = dict(translate_const=int(im_dim * 0.45), img_mean=tuple([min(255, round(255 * channel_mean)) for channel_mean in mean]))
            rand_augmentation_transforms = [rand_augment_transform(aa_config_string, aa_params)]
            # [transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10)]
            re_transform = [transforms.RandomErasing(p=args.reprob, value='random')] if args.reprob > 0.0 else []            
            suffix = suffix +"_AA" if args.basic_augmentation else suffix +"_AAtr0"

        else:
            rand_augmentation_transforms = []
            re_transform = []
            suffix = suffix +"_DA"
            
        transforms_list = train_resize_transforms\
            + rand_augmentation_transforms\
            + basic_data_augmentation_transforms\
            + data_transforms\
            + re_transform

        data_cfg['transform'] = transforms.Compose(transforms_list)    

    return data_cfg, suffix
