import functools
from functools import partial

import torch.nn as nn
import torchvision.transforms as transforms
from timm.models.mlp_mixer import Affine, MlpMixer, ResBlock

### MLP mixer
layers_mlp_cfg = {
    'mixer_s32_224': dict(patch_size=32, num_blocks=8, embed_dim=512),
    'mixer_s16_224': dict(patch_size=16, num_blocks=8, embed_dim=512),
    'mixer_b32_224': dict(patch_size=32, num_blocks=12, embed_dim=768),
    'mixer_b16_224': dict(patch_size=16, num_blocks=12, embed_dim=768),
    'resmlp_12_224': dict(patch_size=16, num_blocks=12, embed_dim=384, mlp_ratio=4, block_layer=ResBlock, norm_layer=Affine),
    'resmlp_24_224': dict(patch_size=16, num_blocks=24, embed_dim=384, mlp_ratio=4, block_layer=partial(ResBlock, init_values=1e-5), norm_layer=Affine),
    'resmlp_36_224': dict(patch_size=16, num_blocks=36, embed_dim=384, mlp_ratio=4,block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine),
    'resmlp_big_24_224': dict(patch_size=8, num_blocks=24, embed_dim=768, mlp_ratio=4, block_layer=partial(ResBlock, init_values=1e-6), norm_layer=Affine),
    'mixer_l32_224': dict(patch_size=32, num_blocks=24, embed_dim=1024),
    'mixer_l16_224': dict(patch_size=16, num_blocks=24, embed_dim=1024),
}

default_mlp_cfg = {
    'in_channels': 3,
    'num_classes': 10,
    'im_dim': (32,32),
    'resize': True,
    'dropout': 0.0,
    'drop_path': 0.1,
    'model_name': 'mixer_s16_224',
    'layers': layers_mlp_cfg['mixer_s16_224']
}

class MLP(nn.Module):
    def __init__(self, model_cfg=default_mlp_cfg, init_weights=True) -> None:
        super(MLP, self).__init__()
        num_classes = model_cfg.get('num_classes',default_mlp_cfg['num_classes'])
        in_channels = model_cfg.get('in_channels',default_mlp_cfg['in_channels'])
        im_dim = model_cfg.get('im_dim',default_mlp_cfg['im_dim'])
        resize = model_cfg.get('resize',default_mlp_cfg['resize'])
        if resize:
            im_dim = (224,224)
            self.resize = transforms.Resize(im_dim)



        layers = model_cfg.get('layers',default_mlp_cfg['layers'])
        model_name = model_cfg.get('model_name',default_mlp_cfg['model_name'])

        if 'dropout' in layers.keys():
            dropout = layers.pop('dropout')
        else:
            dropout = model_cfg.get('dropout',default_mlp_cfg['dropout'])
        if 'drop_path' in layers.keys():
            drop_path = layers.pop('drop_path')
        else:
            drop_path = model_cfg.get('drop_path',default_mlp_cfg['drop_path'])

        self.mlp_model = MlpMixer(img_size=im_dim, in_chans=in_channels, num_classes=num_classes,\
                drop_rate=dropout, drop_path_rate= drop_path,\
                **layers)

    def forward(self,x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        return self.mlp_model.forward(x)

    def forward_features(self,x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        return self.mlp_model.forward_features(x)
