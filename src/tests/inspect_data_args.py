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

import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt 
def imshow(dataset,images,labels,ax1=plt,ax2=None):

    #print("Warning: imshow asumes simple data normalization. Visual artifacts possible for other transforms")
    with torch.no_grad():
        unnormalize = transforms.Normalize(-dataset.mean/dataset.std,1.0/dataset.std)
        images = unnormalize(images)# unnormalize
        img = torchvision.utils.make_grid(images,padding=1,pad_value=1)
        npimg = img.numpy()
        try:
            assert(npimg.min()>-1e-5 and npimg.max()<1+1e-5)
        except AssertionError as e:
            print("Warning: npimg after unnormalizing range (%0.4g,%0.4g)" %(npimg.min(),npimg.max()))
        
        ax1.imshow(np.transpose(npimg.clip(min=0,max=1.0), (1, 2, 0)))
        ax1.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
        
    if ax2 is not None:
        label_text = "    ".join(list(data.classes[labels]))
        ax2.text(0.5,0.5, f'Labels:    {label_text}', horizontalalignment='center', verticalalignment='center', fontsize=24)
        ax2.axis('off')
        ax2.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)


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
        
    data_cfg['root'] = os.environ.get('DATA_DIR', default_cfg[dataset]['root'])
        
    print("Data cfg:", data_cfg)
    data = Dataset(data_cfg = data_cfg)
    trainloader = torch.utils.data.DataLoader(
        data.trainset, shuffle=True,
        batch_size=8,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    f = plt.figure(figsize=(32,8))
    f1 = plt.subplot(2,1,1)    
    f2 = plt.subplot(2,1,2)
    imshow(data,images,labels,f1,f2)

    f.tight_layout()
    f.savefig('samples.pdf',bbox_inches='tight')

    print("Saved sample data to samples.pdf! ")
