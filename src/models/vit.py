import functools

import torch.nn as nn
import torchvision.transforms as transforms
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer

### ViT
layers_vit_cfg = {
    'vit_tiny':{'embed_dim':192, 'depth':12, 'num_heads':3},
    'vit_small':{'embed_dim':384, 'depth':12, 'num_heads':6},
    'vit_base':{'embed_dim':768, 'depth':12, 'num_heads':12},
    'vit_large': {'embed_dim':1024, 'depth':24, 'num_heads':16},
    'vit_huge': {'patch_size':14, 'embed_dim':1280, 'depth':32, 'num_heads':16},
    'cait_xxs24': dict(embed_dim=192, depth=24, num_heads=4, init_values=1e-5, drop_path=0.05),
    'cait_xxs36': dict(embed_dim=192, depth=36, num_heads=4, init_values=1e-5),
    'cait_xs24': dict(embed_dim=288, depth=24, num_heads=6, init_values=1e-5, drop_path=0.05),
    'cait_s24': dict(embed_dim=384, depth=24, num_heads=8, init_values=1e-5),
    'cait_s36': dict(embed_dim=384, depth=36, num_heads=8, init_values=1e-6),
    'cait_m36': dict(embed_dim=768, depth=36, num_heads=16, init_values=1e-6),
    'cait_m48': dict(embed_dim=768, depth=48, num_heads=16, init_values=1e-6),
}

default_vit_cfg = {
    'in_channels': 3,
    'num_classes': 10,
    'im_dim': (32,32),
    'resize': True,
    'dropout': 0.0,
    'drop_path': 0.1,
    'model_name': 'vit_tiny',
    'layers': layers_vit_cfg['vit_tiny']
}

class ViT(nn.Module):
    def __init__(self, model_cfg=default_vit_cfg, init_weights=True) -> None:
        super(ViT, self).__init__()
        num_classes = model_cfg.get('num_classes',default_vit_cfg['num_classes'])
        in_channels = model_cfg.get('in_channels',default_vit_cfg['in_channels'])
        im_dim = model_cfg.get('im_dim',default_vit_cfg['im_dim'])
        resize = model_cfg.get('resize',default_vit_cfg['resize'])
        if resize:
            im_dim = (224,224)
            self.resize = transforms.Resize(im_dim)



        layers = model_cfg.get('layers',default_vit_cfg['layers'])
        model_name = model_cfg.get('model_name',default_vit_cfg['model_name'])

        if 'dropout' in layers.keys():
            dropout = layers.pop('dropout')
        else:
            dropout = model_cfg.get('dropout',default_vit_cfg['dropout'])
        if 'drop_path' in layers.keys():
            drop_path = layers.pop('drop_path')
        else:
            drop_path = model_cfg.get('drop_path',default_vit_cfg['drop_path'])

        if model_name.startswith('vit') or model_name.startswith('deit'):
            self.vit_model = VisionTransformer(img_size=im_dim, in_chans=in_channels, num_classes=num_classes,\
                drop_rate=dropout, attn_drop_rate=dropout, drop_path_rate= drop_path,\
                **layers)
        elif model_name.startswith('cait'):
            self.vit_model = Cait(img_size=im_dim, in_chans=in_channels, num_classes=num_classes,\
                drop_rate=dropout, attn_drop_rate=dropout, drop_path_rate= drop_path,\
                **layers)

    def forward(self,x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        return self.vit_model.forward(x)

    def forward_features(self,x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        return self.vit_model.forward_features(x)

## vit defaults
# img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
# num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
# drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None --> partial(nn.LayerNorm, eps=1e-6),
# act_layer=None --> nn.GELU, weight_init=''

## cait defaults
# img_size=224, patch_size=16, in_chans=3, num_classes=1000,
# embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
# drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
# norm_layer=partial(nn.LayerNorm, eps=1e-6),
# global_pool=None,
# block_layers=LayerScaleBlock,
# block_layers_token=LayerScaleBlockClassAttn,
# patch_layer=PatchEmbed,
# act_layer=nn.GELU,
# attn_block=TalkingHeadAttn,
# mlp_block=Mlp,
# init_scale=1e-4,
# attn_block_token_only=ClassAttn,
# mlp_block_token_only=Mlp,
# depth_token_only=2,
# mlp_ratio_clstk=4.0
