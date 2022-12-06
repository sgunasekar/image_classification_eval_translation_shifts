import functools

import torch.nn as nn
import torchvision.transforms as transforms


class MissingConfiguration(Exception):
    """Required key is missing from the cfg dictionary"""
    pass


layers_vgg_cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M','F',512,512],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M','F',512,512],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M','F',512,512],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M','F',512,512],
}

default_vgg_cfg = {
        'in_channels': 3,
        'num_classes': 10,
        'im_dim': (32,32),
        'padding_mode': 'zeros',
        'batchnorm': True,
        'dropout': 0.5,
        'resize': True,
        'layers': layers_vgg_cfg['vgg16']
        # layer-type=conv at beginning ('M' is max pool), switches to fc after 'F'
}

class VGG(nn.Module):

    def __init__(self, model_cfg=default_vgg_cfg, init_weights=True):

        super(VGG, self).__init__()

        num_classes = model_cfg.get('num_classes',default_vgg_cfg['num_classes'])
        self.features, num_features = self._make_feature_layers(model_cfg)
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

    def forward_features(self,x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        x = self.features(x)
        return x

    def forward(self, x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _make_feature_layers(self,model_cfg):

        #Default configuration
        in_channels = model_cfg.get('in_channels',default_vgg_cfg['in_channels'])
        im_dim = model_cfg.get('im_dim',default_vgg_cfg['im_dim'])
        if isinstance(im_dim,int):
            im_dim = (im_dim,im_dim)
        resize = model_cfg.get('resize',default_vgg_cfg['resize'])
        if resize:
            im_dim = (224,224)
            self.resize = transforms.Resize(im_dim)

        batchnorm = model_cfg.get('batchnorm',default_vgg_cfg['batchnorm'])

        dropout = model_cfg.get('dropout',default_vgg_cfg['dropout'])
        padding_mode = model_cfg.get('padding_mode',default_vgg_cfg['padding_mode'])

        if 'layers' not in model_cfg.keys():
            #[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M','F',512,512]
            raise MissingConfiguration("Required key 'layers' is missing in model_cfg")

        print("Model: ", model_cfg['layers'], " batchnorm: ", batchnorm, " dropout: ",dropout, " in_channels: ", in_channels, " im_dim: ", im_dim, " padding_mode: ", padding_mode)

        layers = []
        layer_type = 'conv'

        for v in model_cfg['layers']:
            if v == 'M':
                kersize = 2
                if im_dim[0]==3:
                    kersize = 3
                elif im_dim[0]==1:
                    kersize = 1
                layers += [nn.MaxPool2d(kernel_size=kersize, stride=2)]
                im_dim = (im_dim[0]//kersize,im_dim[1]//kersize)
            elif v == 'A':
                layers += [nn.AdaptiveAvgPool2d((7,7))]
                im_dim = (7,7)
            elif v=='F':
                layer_type='fc'
                layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
                im_dim = (1,1)
                num_features = in_channels
            elif v=='C': #unused
                layer_type='conv'
            else:
                if layer_type=='conv':
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, padding_mode=padding_mode,bias=(not batchnorm))
                    if batchnorm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
                elif layer_type=='fc':
                    if dropout>0:
                        layers += [nn.Dropout(p=dropout), nn.Linear(num_features, v), nn.ReLU(inplace=True)]
                    else:
                        layers += [nn.Linear(num_features, v), nn.ReLU(inplace=True)]
                    num_features = v
                else:
                    print('Warning: "layers" element %s is not understood. Skipping.' %v)

        if layer_type=='fc':
            num_features = num_features
        elif layer_type=='conv':
            layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
            im_dim = (1,1)
            num_features = in_channels

        return nn.Sequential(*layers),num_features
