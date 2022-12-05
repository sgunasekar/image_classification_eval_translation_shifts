import functools

import antialiased_cnns
import torch.nn as nn
import torchvision.transforms as transforms

layers_antialiased_resnet_cfg = {
    'antialiased_resnet18':antialiased_cnns.resnet18, #{'block':BasicBlock, 'num_blocks':[2, 2, 2, 2], 'filter_size':4},
    'antialiased_resnet34':antialiased_cnns.resnet34, #{'block':BasicBlock, 'num_blocks':[3, 4, 6, 3], 'filter_size':4},
    'antialiased_resnet50':antialiased_cnns.resnet50, #{'block':Bottleneck, 'num_blocks':[3, 4, 6, 3], 'filter_size':4},
    'antialiased_resnet101':antialiased_cnns.resnet101, #{'block':Bottleneck, 'num_blocks':[3, 4, 23, 3], 'filter_size':4},
    'antialiased_resnet152':antialiased_cnns.resnet152, #{'block':Bottleneck, 'num_blocks':[3, 8, 36, 3], 'filter_size':4},
    'antialiased_resnext50_32x4d':antialiased_cnns.resnext50_32x4d, #{'block':Bottleneck, 'num_blocks':[3, 4, 6, 3], 'groups':32, 'width_per_group':4, 'filter_size':4},
    'antialiased_resnext101_32x8d':antialiased_cnns.resnext101_32x8d, #{'block':Bottleneck, 'num_blocks':[3,4,23,3], 'base_channel':64, 'groups':32, 'base_width_per_group':8, 'filter_size':4},
    'antialiased_wide_resnet50_2':antialiased_cnns.wide_resnet50_2, #{'block':Bottleneck, 'num_blocks':[3,4,6,3], 'base_channel':64, 'scale_width': 2, 'filter_size':4},
    'antialiased_wide_resnet101_2':antialiased_cnns.wide_resnet101_2, #{'block':Bottleneck, 'num_blocks':[3,4,23,3], 'base_channel':64, 'scale_width': 2, 'filter_size':4},
}
default_antialiased_resnet_cfg = {
    'in_channels': 3,
    'num_classes': 10,
    'im_dim': (32,32),
    'layers': layers_antialiased_resnet_cfg['antialiased_resnet18'],
    'resize': True,
    'filter_size': 4,
}

class AntiAliased_ResNet(nn.Module):
    def __init__(self, model_cfg=default_antialiased_resnet_cfg, init_weights=True) -> None:
        
        super(AntiAliased_ResNet, self).__init__()
        
        num_classes = model_cfg.get('num_classes',default_antialiased_resnet_cfg['num_classes'])
        in_channels = model_cfg.get('in_channels',default_antialiased_resnet_cfg['in_channels'])
        filter_size = model_cfg.get('filter_size',default_antialiased_resnet_cfg['filter_size'])
        
        im_dim = model_cfg.get('im_dim',default_antialiased_resnet_cfg['im_dim'])
        if isinstance(im_dim,int):
            im_dim=(im_dim,im_dim)
        resize = model_cfg.get('resize',default_antialiased_resnet_cfg['resize'])
        if resize:
            im_dim = (224,224)
            self.resize = transforms.Resize(im_dim)
        
        model_class = model_cfg.get('layers',default_antialiased_resnet_cfg['layers'])
        
        self.model = model_class(filter_size=filter_size,num_classes=num_classes)
        
        if im_dim[0]<128:
            self.model.conv1 = nn.Conv2d(in_channels, self.model.inplanes, kernel_size=3, padding=1, stride=1, bias = False)
        
    def forward(self,x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        return self.model.forward(x)
