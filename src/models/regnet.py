## RegNet (regnetx,regnety)
# Adapted from timm

import functools

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from timm.models.layers import (AvgPool2dSame, ClassifierHead, DropPath,
                                create_act_layer, get_norm_act_layer)
from timm.models.layers.helpers import make_divisible
from timm.models.layers.padding import get_padding_value

from .resnet import StdConv2d


class GroupNormPartialWrapper(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        super(GroupNormPartialWrapper, self).__init__(num_groups, num_channels, eps=eps, affine=affine)

 # Adapting ConvBNAct and SEModule in timm to work with weight standardization
 # Norm layer currently can only be functools.partial(nn.GroupNormPartialWrapper(num_groups=32)) or nn.BatchNorm2d

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1, bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, drop_block=None):

        super(ConvBnAct, self).__init__()

        groupnorm = isinstance(norm_layer, functools.partial) and (norm_layer.func.__name__.lower().startswith('groupnorm'))
        padding, _ = get_padding_value(padding, kernel_size, stride=stride, dilation = dilation)
        if groupnorm:
            self.conv = StdConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias = bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias = bias)


        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        if norm_layer is not None:
            norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
            self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block)
        else:
            self.bn = create_act_layer(act_layer,inplace=True)
        self.bn = self.bn or nn.Identity()

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False, act_layer=nn.ReLU, norm_layer=None, gate_layer='sigmoid'):

        super(SEModule, self).__init__()

        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)

        groupnorm = isinstance(norm_layer, functools.partial) and (norm_layer.func.__name__.lower().startswith('groupnorm'))

        if groupnorm:
            self.fc1 = StdConv2d(channels, rd_channels, kernel_size=1, bias=True)
        else:
            self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)

        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()

        self.act = create_act_layer(act_layer, inplace=True)

        if groupnorm:
            self.fc2 = StdConv2d(rd_channels, channels, kernel_size=1, bias=True)
        else:
            self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)

        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

## RegNet Code
def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)

def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
    return widths, num_stages, max_stage, widths_cont

class Bottleneck(nn.Module):
    """ RegNet Bottleneck
    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, bottleneck_ratio=1, group_width=1, se_ratio=0.25, downsample=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_block=None, drop_path=None):

        super(Bottleneck, self).__init__()
        bottleneck_chs = int(round(out_chs * bottleneck_ratio))
        groups = bottleneck_chs // group_width

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer, drop_block=drop_block)

        self.conv1 = ConvBnAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)

        self.conv2 = ConvBnAct(bottleneck_chs, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation, groups=groups, **cargs)

        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels)
        else:
            self.se = None

        cargs['act_layer'] = None
        self.conv3 = ConvBnAct(bottleneck_chs, out_chs, kernel_size=1, **cargs)

        self.act3 = act_layer(inplace=True)

        self.downsample = downsample

        self.drop_path = drop_path

    def zero_init_last_bn(self):
        if self.conv3.bn is not None :
            nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        return x


def downsample_conv(in_chs, out_chs, kernel_size, stride=1, dilation=1, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    return ConvBnAct(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, norm_layer=norm_layer, act_layer=None)


def downsample_avg(in_chs, out_chs, kernel_size, stride=1, dilation=1, norm_layer=None):
    """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    pool = nn.Identity()
    if stride > 1 or dilation > 1:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    return nn.Sequential(*[pool, ConvBnAct(in_chs, out_chs, 1, stride=1, norm_layer=norm_layer, act_layer=None)])


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio, group_width,
                 block_fn=Bottleneck, se_ratio=0., drop_path_rates=None, drop_block=None, norm_layer=None):

        super(RegStage, self).__init__()
        block_kwargs = {}  # FIXME setup to pass various aa,  act layer common args
        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = first_dilation if i == 0 else dilation
            if drop_path_rates is not None and drop_path_rates[i] > 0.:
                drop_path = DropPath(drop_path_rates[i])
            else:
                drop_path = None
            if (block_in_chs != out_chs) or (block_stride != 1):
                proj_block = downsample_conv(block_in_chs, out_chs, 1, block_stride, block_dilation, norm_layer)
            else:
                proj_block = None

            name = "b{}".format(i + 1)
            self.add_module(
                name, block_fn(block_in_chs, out_chs, block_stride, block_dilation, bottle_ratio, group_width, se_ratio, downsample=proj_block, drop_block=drop_block, drop_path=drop_path, norm_layer=norm_layer, **block_kwargs)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

def _mcfg(**kwargs):
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32, output_stride=32, zero_init_last_bn=True, global_pool='avg')
    cfg.update(**kwargs)
    return cfg

layers_regnet_cfg = dict(
    regnetx_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13),
    regnetx_004=_mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22),
    regnetx_006=_mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16),
    regnetx_008=_mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16),
    regnetx_016=_mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    regnetx_032=_mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25),
    regnetx_040=_mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23),
    regnetx_064=_mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17),
    regnetx_080=_mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23),
    regnetx_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19),
    regnetx_160=_mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22),
    regnetx_320=_mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23),
    regnety_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    regnety_004=_mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25),
    regnety_006=_mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    regnety_008=_mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),
    regnety_016=_mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    regnety_032=_mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    regnety_040=_mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    regnety_064=_mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    regnety_080=_mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),
    regnety_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    regnety_160=_mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    regnety_320=_mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25),
    regnetxy_002=_mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    regnetxy_004=_mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22, se_ratio=0.25),
    regnetxy_006=_mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16, se_ratio=0.25),
    regnetxy_008=_mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16, se_ratio=0.25),
    regnetxy_016=_mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18, se_ratio=0.25),
    regnetxy_032=_mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25, se_ratio=0.25),
    regnetxy_040=_mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23, se_ratio=0.25),
    regnetxy_064=_mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17, se_ratio=0.25),
    regnetxy_080=_mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23, se_ratio=0.25),
    regnetxy_120=_mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    regnetxy_160=_mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22, se_ratio=0.25),
    regnetxy_320=_mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23, se_ratio=0.25),
)


default_regnet_cfg = {
    'in_channels': 3,
    'num_classes': 10,
    'im_dim': (32,32),
    'batchnorm': False,
    'groupnorm': False,
    'gn_num_groups': 32,
    'dropout': 0,
    'drop_path': 0,
    'layers': layers_regnet_cfg['regnety_040'],
    'resize': False
    #{'se_ratio':0, 'bottle_ratio':1, 'stem_width':32, 'w0':200, 'wa':106.23, 'wm':2.48,\
    #    'group_w':112, 'depth':18, 'se_ratio':0.25}
}

class RegNet(nn.Module):
    """RegNet model.
    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(self, model_cfg=default_regnet_cfg, init_weights=True) -> None:

        super(RegNet, self).__init__()

        num_classes = model_cfg.get('num_classes',default_regnet_cfg['num_classes'])
        in_channels = model_cfg.get('in_channels',default_regnet_cfg['in_channels'])
        im_dim = model_cfg.get('im_dim',default_regnet_cfg['im_dim'])

        if isinstance(im_dim,list):
            self.im_dim=im_dim
        resize = model_cfg.get('resize',default_regnet_cfg['resize'])
        if resize:
            im_dim = 224
            self.resize = transforms.Resize(im_dim)
        self.input_size = (in_channels, im_dim, im_dim)

        dropout = model_cfg.get('dropout',default_regnet_cfg['dropout'])
        drop_path = model_cfg.get('drop_path',default_regnet_cfg['drop_path'])

        batchnorm = model_cfg.get('batchnorm',default_regnet_cfg['batchnorm'])
        groupnorm = model_cfg.get('groupnorm',default_regnet_cfg['groupnorm'])
        if groupnorm:
            gn_num_groups = model_cfg.get('gn_num_groups', default_regnet_cfg['gn_num_groups'])
            norm_layer = functools.partial(GroupNormPartialWrapper,num_groups = gn_num_groups)
        elif batchnorm:
            norm_layer = nn.BatchNorm2d
        else:
            print("Warning: No normalization is used")
            norm_layer = None

        layers = model_cfg.get('layers',default_regnet_cfg['layers'])


        # Construct the stem
        stem_width = layers['stem_width']
        output_stride = layers['output_stride']
        assert output_stride in (8, 16, 32)

        # Changed stride of stem for cifar to 1
        stem_stride = 2 if resize else 1
        self.stem = ConvBnAct(in_channels, stem_width, 3, stride=stem_stride, norm_layer = norm_layer)
        self.feature_info = [dict(num_chs=stem_width, reduction=stem_stride, module='stem')]

        # Construct the stages
        prev_width = stem_width
        curr_stride = stem_stride
        stage_params = self._get_stage_params(layers, output_stride=output_stride, drop_path_rate=drop_path)
        se_ratio = layers['se_ratio']
        for i, stage_args in enumerate(stage_params):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(prev_width, **stage_args, se_ratio=se_ratio, norm_layer=norm_layer))
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]

        # Construct the head
        global_pool = layers.get('global_pool','avg')
        zero_init_last_bn = layers.get('zero_init_last_bn','True') and (norm_layer is not None)
        self.num_features = prev_width
        self.head = ClassifierHead(
            in_chs=prev_width, num_classes=num_classes, pool_type=global_pool, drop_rate=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _get_stage_params(self, cfg, default_stride=2, output_stride=32, drop_path_rate=0.):
        # Generate RegNet ws per block
        w_a, w_0, w_m, d = cfg['wa'], cfg['w0'], cfg['wm'], cfg['depth']
        widths, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)

        # Convert to per stage format
        stage_widths, stage_depths = np.unique(widths, return_counts=True)

        # Use the same group width, bottleneck mult and stride for each stage
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_bottle_ratios = [cfg['bottle_ratio'] for _ in range(num_stages)]
        stage_strides = []
        stage_dilations = []
        net_stride = self.feature_info[-1]['reduction']

        for i in range(num_stages):
            dilation = 1
            if not hasattr(self,'resize') and i==0:
                stride = 1
                net_stride *= stride
            elif net_stride >= output_stride:
                dilation *= default_stride
                stride = 1
            else:
                stride = default_stride
                net_stride *= stride

            stage_strides.append(stride)
            stage_dilations.append(dilation)
        stage_dpr = np.split(np.linspace(0, drop_path_rate, d), np.cumsum(stage_depths[:-1]))

        # Adjust the compatibility of ws and gws
        stage_widths, stage_groups = adjust_widths_groups_comp(stage_widths, stage_bottle_ratios, stage_groups)
        param_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_width', 'drop_path_rates']
        stage_params = [
            dict(zip(param_names, params)) for params in
            zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_bottle_ratios, stage_groups,
                stage_dpr)]
        print("Stage parameters")
        print(" width:", stage_widths, "\n depths:", stage_depths, "\n strides:", stage_strides, "\n dilations:", stage_dilations)
        print(" bottleneck ratios:", stage_bottle_ratios, "\n groups:", stage_groups)
        return stage_params

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        for block in list(self.children())[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        if hasattr(self,'resize'):
            x = self.resize(x)
        for block in self.children():
            x = block(x)
        return x
