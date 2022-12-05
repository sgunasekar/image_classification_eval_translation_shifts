import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, bias = self.bias, stride = self.stride, padding = self.padding, dilation = self.dilation, groups = self.groups)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool = True, weight_standardization: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    if not weight_standardization:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, dilation=dilation, bias = bias)
    else:
        return StdConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, dilation=dilation, bias = bias)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = True,  weight_standardization: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    if not weight_standardization:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return StdConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


### ResNet (resnet, resnext, wide_resnet, preact_resnet)

class BasicBlock(nn.Module):

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        batchnorm: bool= True,
        groupnorm: bool= False,
        gn_num_groups = 32,
    ) -> None:

        super(BasicBlock, self).__init__()

        normalization = (batchnorm or groupnorm)
        if batchnorm and groupnorm:
            print('Warning: batchnorm argument is ignored when groupnorm is set to true')

        conv1 = conv3x3(in_channels, out_channels, stride=stride, bias=(not normalization), weight_standardization = groupnorm)
        conv2 = conv3x3(out_channels, out_channels, bias=(not normalization), weight_standardization = groupnorm)

        if groupnorm:
            gn1 = nn.GroupNorm(gn_num_groups, out_channels)
            gn2 = nn.GroupNorm(gn_num_groups, out_channels)
            self.layers = nn.Sequential(*[conv1,gn1,nn.ReLU(inplace=True),conv2,gn2])
        elif batchnorm:
            bn1 = nn.BatchNorm2d(out_channels)
            bn2 = nn.BatchNorm2d(out_channels)
            self.layers = nn.Sequential(*[conv1,bn1,nn.ReLU(inplace=True),conv2,bn2])
        else:
            self.layers = nn.Sequential(*[conv1,nn.ReLU(inplace=True),conv2])

        self.shortcut = []
        if stride != 1 or in_channels != out_channels:
            self.shortcut += [conv1x1(in_channels, out_channels, stride=stride, bias=(not normalization), weight_standardization=groupnorm)]
            if groupnorm:
                self.shortcut += [nn.GroupNorm(gn_num_groups, out_channels)]
            elif batchnorm:
                self.shortcut += [nn.BatchNorm2d(out_channels)]
        self.shortcut = nn.Sequential(*self.shortcut)


    def forward(self, x: Tensor) -> Tensor:

        out = self.layers(x)
        out += self.shortcut(x) #Id or conv_1x1xRxC_stride
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        scale_width: int = 1, #=base_width*group/64 in pytorch code
        stride: int = 1,
        batchnorm: bool= True,
        groupnorm: bool= False,
        gn_num_groups: int = 32
    ) -> None:

        super(Bottleneck, self).__init__()

        normalization = (batchnorm or groupnorm)
        if batchnorm and groupnorm:
            print('Warning: batchnorm argument is ignored when groupnorm is set to true')

        bottleneck_width = int(out_channels*scale_width/self.expansion)

        conv1 = conv1x1(in_channels, bottleneck_width, bias=(not normalization), weight_standardization = groupnorm)
        conv2 = conv3x3(bottleneck_width, bottleneck_width, stride=stride, groups = groups, bias=(not normalization), weight_standardization = groupnorm)
        conv3 = conv1x1(bottleneck_width, out_channels, bias=(not normalization), weight_standardization = groupnorm)

        if groupnorm:
            gn1 = nn.GroupNorm(gn_num_groups, bottleneck_width)
            gn2 = nn.GroupNorm(gn_num_groups, bottleneck_width)
            gn3 = nn.GroupNorm(gn_num_groups, out_channels)
            self.layers = nn.Sequential(*[conv1,gn1,nn.ReLU(inplace=True),conv2,gn2,nn.ReLU(inplace=True),conv3,gn3])
        elif batchnorm:
            bn1 = nn.BatchNorm2d(bottleneck_width)
            bn2 = nn.BatchNorm2d(bottleneck_width)
            bn3 = nn.BatchNorm2d(out_channels)
            self.layers = nn.Sequential(*[conv1,bn1,nn.ReLU(inplace=True),conv2,bn2,nn.ReLU(inplace=True),conv3,bn3])
        else:
            self.layers = nn.Sequential(*[conv1,nn.ReLU(inplace=True),conv2,nn.ReLU(inplace=True), conv3])

        self.shortcut = []
        if stride != 1 or in_channels != out_channels:
            self.shortcut += [conv1x1(in_channels, out_channels, stride=stride, bias=(not normalization), weight_standardization=groupnorm)]

            if groupnorm:
                self.shortcut += [nn.GroupNorm(gn_num_groups, out_channels)]
            elif batchnorm:
                self.shortcut += [nn.BatchNorm2d(out_channels)]
        self.shortcut = nn.Sequential(*self.shortcut)

    def forward(self, x: Tensor) -> Tensor:

        out = self.layers(x)
        out += self.shortcut(x) #Id or conv_1x1xRxC_stride
        out = F.relu(out)
        return out

class PreActBasicBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        batchnorm: bool = True,
        groupnorm: bool = False,
        gn_num_groups: int = 32
    ) -> None:

        super(PreActBasicBlock, self).__init__()

        normalization = (batchnorm or groupnorm)
        if batchnorm and groupnorm:
            print('Warning: batchnorm argument is ignored when groupnorm is set to true')

        conv1 = conv3x3(in_channels, out_channels, stride=stride, bias=(not normalization), weight_standardization=groupnorm)
        conv2 = conv3x3(out_channels, out_channels, bias=(not normalization), weight_standardization=groupnorm)
        relu0 = nn.ReLU()
        relu1 = nn.ReLU()

        if groupnorm:
            gn0 = nn.GroupNorm(gn_num_groups,in_channels)
            gn1 = nn.GroupNorm(gn_num_groups,out_channels)
            self.preact_layer = nn.Sequential(*[gn0,relu0])
            self.layers = nn.Sequential(*[conv1,gn1,relu1,conv2])
        elif batchnorm:
            bn0 = nn.BatchNorm2d(in_channels)
            bn1 = nn.BatchNorm2d(out_channels)
            self.preact_layer = nn.Sequential(*[bn0,relu0])
            self.layers = nn.Sequential(*[conv1,bn1,relu1,conv2])
        else:
            self.preact_layer = nn.Sequential(*[relu0])
            self.layers = nn.Sequential(*[conv1,relu1,conv2])

        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv1x1(in_channels, out_channels, stride=stride, bias=(not normalization), weight_standardization=groupnorm)

    def forward(self, x):
        out = self.preact_layer(x)
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.layers(out)
        out += shortcut #Id or conv_1x1xRxC_stride

        return out

class PreActBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        scale_width: int = 1, #=base_width*group/64 in pytorch code
        stride: int = 1,
        batchnorm: bool = True,
        groupnorm: bool = False,
        gn_num_groups: int = 32
    ) -> None:

        super(PreActBottleneck, self).__init__()

        normalization = (batchnorm or groupnorm)
        if batchnorm and groupnorm:
            print('Warning: batchnorm argument is ignored when groupnorm is set to true')

        bottleneck_width = int(out_channels*scale_width/self.expansion)

        conv1 = conv1x1(in_channels, bottleneck_width, bias=(not normalization), weight_standardization=groupnorm)
        conv2 = conv3x3(bottleneck_width, bottleneck_width, stride=stride, groups = groups, bias=(not normalization), weight_standardization=groupnorm)
        conv3 = conv1x1(bottleneck_width, out_channels, bias=(not normalization),  weight_standardization=groupnorm)

        relu0 = nn.ReLU()
        relu1 = nn.ReLU()
        relu2 = nn.ReLU()

        if groupnorm:
            gn0 = nn.GroupNorm(gn_num_groups,in_channels)
            gn1 = nn.GroupNorm(gn_num_groups,bottleneck_width)
            gn2 = nn.GroupNorm(gn_num_groups,bottleneck_width)
            self.preact_layer = nn.Sequential(*[gn0,relu0])
            self.layers = nn.Sequential(*[conv1,gn1,relu1,\
                                        conv2,gn2,relu2,conv3])

        elif batchnorm:
            bn0 = nn.BatchNorm2d(in_channels)
            bn1 = nn.BatchNorm2d(bottleneck_width)
            bn2 = nn.BatchNorm2d(bottleneck_width)
            self.preact_layer = nn.Sequential(*[bn0,relu0])
            self.layers = nn.Sequential(*[conv1,bn1,relu1,\
                                        conv2,bn2,relu2,conv3])
        else:
            self.preact_layer = nn.Sequential(*[relu0])
            self.layers = nn.Sequential(*[conv1,relu1,\
                                        conv2,relu2,conv3])


        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv1x1(in_channels, out_channels, stride=stride, bias=(not normalization), weight_standardization=groupnorm)

    def forward(self, x: Tensor) -> Tensor:

        out = self.preact_layer(x)
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.layers(out)
        out += shortcut

        return out

layers_resnet_cfg = {
    # Large models in pytorch
    'resnet18': {'block':BasicBlock, 'num_blocks':[2,2,2,2], 'base_channel':64},
    'resnet34': {'block':BasicBlock, 'num_blocks':[3,4,6,3], 'base_channel':64},
    'resnet50': {'block':Bottleneck, 'num_blocks':[3,4,6,3], 'base_channel':64},
    'resnet101': {'block':Bottleneck, 'num_blocks':[3,4,23,3], 'base_channel':64},
    'resnet152': {'block':Bottleneck, 'num_blocks':[3,8,36,3], 'base_channel':64},
    'resnext50_32x4d': {'block':Bottleneck, 'num_blocks':[3,4,6,3], 'base_channel':64, 'groups':32, 'base_width_per_group':4},
    'resnext101_32x8d': {'block':Bottleneck, 'num_blocks':[3,4,23,3], 'base_channel':64, 'groups':32, 'base_width_per_group':8},
    # WRN uses standard activation for Imagenet "For ImageNet, we find that using pre-activation in networks with less than 100 layers does not make any significant difference and so we decide to use the original ResNet architecture in this case"
    'wide_resnet50_2': {'block':Bottleneck, 'num_blocks':[3,4,6,3], 'base_channel':64, 'scale_width': 2},
    'wide_resnet101_2': {'block':Bottleneck, 'num_blocks':[3,4,23,3], 'base_channel':64, 'scale_width': 2},
    # Small models
    # Models in papers for CIFAR10
    'resnet20': {'block':BasicBlock, 'num_blocks':[3,3,3,0], 'base_channel':16},  # te 8.75
    'resnet32': {'block':BasicBlock, 'num_blocks':[5,5,5,0], 'base_channel':16}, # te 7.51
    'resnet44': {'block':BasicBlock, 'num_blocks':[7,7,7,0], 'base_channel':16}, # te 7.17
    'resnet56': {'block':BasicBlock, 'num_blocks':[9,9,9,0], 'base_channel':16}, # te 6.97
    'resnet110': {'block':BasicBlock, 'num_blocks':[18,18,18,0], 'base_channel':16}, # te 6.43
    'resnet1202': {'block':BasicBlock, 'num_blocks':[200,200,200,0], 'base_channel':16}, # te 7.93
    'resnext29_2x64d': {'block':Bottleneck, 'num_blocks':[3,3,3,0], 'base_channel':64, 'groups':2, 'base_width_per_group':64},
    'resnext29_4x64d': {'block':Bottleneck, 'num_blocks':[3,3,3,0], 'base_channel':64, 'groups':4, 'base_width_per_group':64},
    'resnext29_8x64d': {'block':Bottleneck, 'num_blocks':[3,3,3,0], 'base_channel':64, 'groups':8, 'base_width_per_group':64}, #3.65
    'resnext29_16x64d': {'block':Bottleneck, 'num_blocks':[3,3,3,0], 'base_channel':64, 'groups':16, 'base_width_per_group':64},#3.58
    'resnext29_32x4d': {'block':Bottleneck, 'num_blocks':[3,3,3,0], 'base_channel':64, 'groups':32, 'base_width_per_group':4},
    # In WRN paper to depth is counted differently (includes 1x1 shortcut convolutions but excludes classifier layer, so wide_resent40_4 in paper is wide_resent38_4 etc.
    # WRN was also implemented only with preact
    'wide_resnet38_8': {'block':BasicBlock, 'num_blocks':[12,12,12,0], 'base_channel':16, 'scale_width':8},
    'wide_resnet26_10': {'block':BasicBlock, 'num_blocks':[8,8,8,0], 'base_channel':16, 'scale_width':10},
    'wide_resnet20_10': {'block':BasicBlock, 'num_blocks':[4,4,4,0], 'base_channel':16, 'scale_width':10},
    'wide_resnet14_10': {'block':BasicBlock, 'num_blocks':[4,4,4,0], 'base_channel':16, 'scale_width':10},
    # Preact blocks
    'preact_resnet18': {'block':PreActBasicBlock, 'num_blocks':[2,2,2,2], 'base_channel':64},
    'preact_resnet50': {'block':PreActBottleneck, 'num_blocks':[3,4,6,3], 'base_channel':64},
    'preact_wide_resnet38_8': {'block':PreActBasicBlock, 'num_blocks':[6,6,6,0], 'base_channel':16, 'scale_width':8},
    'preact_wide_resnet26_4': {'block':PreActBasicBlock, 'num_blocks':[4,4,4,0], 'base_channel':16, 'scale_width':4},
    'preact_wide_resnet20_4': {'block':PreActBasicBlock, 'num_blocks':[3,3,3,0], 'base_channel':16, 'scale_width':4},
    'preact_wide_resnet14_4': {'block':PreActBasicBlock, 'num_blocks':[2,2,2,0], 'base_channel':16, 'scale_width':4},
}



default_resnet_cfg = {
    'in_channels': 3,
    'num_classes': 10,
    'im_dim': (32,32),
    'batchnorm': False,
    'groupnorm': False,
    'gn_num_groups': 32,
    'layers': layers_resnet_cfg['resnet18'],
    'resize': True
}
class ResNet(nn.Module):

    def __init__(self, model_cfg=default_resnet_cfg, init_weights=True) -> None:

        super(ResNet, self).__init__()

        num_classes = model_cfg.get('num_classes',default_resnet_cfg['num_classes'])
        self.in_channels = model_cfg.get('in_channels',default_resnet_cfg['in_channels'])

        im_dim = model_cfg.get('im_dim',default_resnet_cfg['im_dim'])
        if isinstance(im_dim,int):
            im_dim=(im_dim,im_dim)
        resize = model_cfg.get('resize',default_resnet_cfg['resize'])
        if resize:
            im_dim = (224,224)
            self.resize = transforms.Resize(im_dim)

        batchnorm = model_cfg.get('batchnorm',default_resnet_cfg['batchnorm'])
        groupnorm = model_cfg.get('groupnorm',default_resnet_cfg['groupnorm'])
        if groupnorm: gn_num_groups = model_cfg.get('gn_num_groups', default_resnet_cfg['gn_num_groups'])

        layers = model_cfg.get('layers',default_resnet_cfg['layers'])

        base_channel = layers.get('base_channel',default_resnet_cfg['layers']['base_channel'])
        block = layers.get('block',default_resnet_cfg['layers']['block'])
        num_blocks = layers.get('num_blocks',default_resnet_cfg['layers']['num_blocks'])
        scale_width = layers.get('scale_width',1)


        out_channels = [base_channel*scale_width*block.expansion*(2**stage) for stage in range(4)]
        stride = [1,2,2,2]
        normalization = (batchnorm or groupnorm)

        kwargs={}
        if 'groups' in layers.keys() or 'base_width_per_group' in layers.keys():
            groups = layers.get('groups',1)
            base_width_per_group = layers.get('base_width_per_group',base_channel)
            scale_width = base_width_per_group*groups/base_channel
            kwargs['groups'] = groups
            kwargs['scale_width'] = scale_width
        if groupnorm and 'gn_num_groups' in model_cfg.keys():
            kwargs['gn_num_groups']=gn_num_groups

        ## Layers
        base_conv_kernel = 7 if im_dim[0]>128 else 3
        base_conv_padding = 3 if im_dim[0]>128 else 1
        base_conv_stride = 1
        if groupnorm:
            conv0 = StdConv2d(self.in_channels, base_channel, kernel_size=base_conv_kernel, padding=base_conv_padding, stride=base_conv_stride, bias = not normalization)
        else:
            conv0 = nn.Conv2d(self.in_channels, base_channel, kernel_size=base_conv_kernel, padding=base_conv_padding, stride=base_conv_stride, bias = not normalization)

        if block==PreActBasicBlock or block==PreActBottleneck:
            self.base_layer = [conv0]
        elif groupnorm:
            gn0 = nn.GroupNorm(gn_num_groups, base_channel)
            self.base_layer = [conv0,gn0,nn.ReLU(inplace=True)]
        elif batchnorm:
            bn0 = nn.BatchNorm2d(base_channel)
            self.base_layer = [conv0,bn0,nn.ReLU(inplace=True)]
        else:
            self.base_layer = [conv0,nn.ReLU(inplace=True)]

        if resize:
            # In maxpool, if padding is non-zero, then the input is implicitly padded with negative infinity
            # Doesnt matter with conv,norm,relu pipeline as 0=min value, but makes a difference for preact blocks
            pad0 = nn.ConstantPad2d(1,0)
            pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            self.base_layer = self.base_layer + [pad0, pool0]


        self.base_layer = nn.Sequential(*self.base_layer)
        self.in_channels = base_channel


        self.stage1 = self._make_stage(block=block, out_channels=out_channels[0], num_blocks=num_blocks[0], stride=stride[0],batchnorm=batchnorm,groupnorm=groupnorm,kwargs=kwargs)
        self.stage2 = self._make_stage(block=block, out_channels=out_channels[1], num_blocks=num_blocks[1], stride=stride[1],batchnorm=batchnorm,groupnorm=groupnorm,kwargs=kwargs)
        self.stage3 = self._make_stage(block=block, out_channels=out_channels[2], num_blocks=num_blocks[2], stride=stride[2],batchnorm=batchnorm,groupnorm=groupnorm,kwargs=kwargs)
        self.stage4 = self._make_stage(block=block, out_channels=out_channels[3], num_blocks=num_blocks[3], stride=stride[3],batchnorm=batchnorm,groupnorm=groupnorm,kwargs=kwargs)
        #im_dim = (im_dim[0]//stride[stage],im_dim[1]//stride[stage])


        if block!=PreActBasicBlock and block!=PreActBottleneck:
            self.pool_layer = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()])
        elif groupnorm:
            gn0 = nn.GroupNorm(gn_num_groups, self.in_channels)
            self.pool_layer = nn.Sequential(*[gn0,nn.ReLU(inplace=True),\
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()])
        elif batchnorm:
            bn0 = nn.BatchNorm2d(self.in_channels)
            self.pool_layer = nn.Sequential(*[bn0,nn.ReLU(inplace=True),\
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()])
        else:
            self.pool_layer = nn.Sequential(*[nn.ReLU(inplace=True),\
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()])

        num_features = self.in_channels

        self.classifier = nn.Linear(num_features, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        x = self.base_layer(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool_layer(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

    def _make_stage(self,block,out_channels,num_blocks,stride,batchnorm,groupnorm,kwargs):
        if num_blocks<1:
            #print("Warning: no. of blocks<1 %d. No block included." %num_blocks)
            return nn.Sequential()

        strides = [stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layer = block(in_channels = self.in_channels, out_channels = out_channels, stride = stride,batchnorm = batchnorm, groupnorm=groupnorm,**kwargs)
            layers.append(layer)
            self.in_channels = out_channels
        return nn.Sequential(*layers)
