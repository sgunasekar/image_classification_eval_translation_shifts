from .config import *
import logging

vgg_prefixes = ("vgg")
mlp_prefixes = ("mixer","resmlp")
resnet_prefixes = ("resnet", "resnext", "wide_resnet", "preact")
resnet_bit_prefixes = ("BiT")
antialiased_resnet_prefixes = ("antialiased_resnet", "antialiased_resnext", "antialiased_wide_resnet")
vit_prefixes = ("deit", "vit", "cait")

def get_model_cfg(args):

    if isinstance(args,str):
        model_name = args
    else:
        model_name = args.model

    if model_name.startswith(vgg_prefixes):
        
        from models.vgg import VGG, default_vgg_cfg, layers_vgg_cfg 
        Net=VGG
        model_cfg = default_vgg_cfg.copy()
        layers_cfg = layers_vgg_cfg

        if model_name in layers_cfg.keys():
            model_cfg['layers'] = layers_cfg[model_name]
        else:
            logging.warning(f"Model name {model_name} is not a valid key for layers_cfg. Using default vgg16 layer config")
            model_name = 'vgg16'

        model_cfg['model_name'] = model_name

    elif model_name.startswith(resnet_prefixes):
        
        from models.resnet import ResNet, default_resnet_cfg, layers_resnet_cfg 
        Net = ResNet
        model_cfg = default_resnet_cfg.copy()
        layers_cfg = layers_resnet_cfg

        if model_name in layers_cfg.keys():
            model_cfg['layers']=layers_cfg[model_name]
        else:
            logging.warning(f"Model name {model_name} is not a valid key for layers_cfg. Using default resnet18 layer config")
            model_name = 'resnet18'

        model_cfg['model_name'] = model_name
    
    elif model_name.startswith(resnet_bit_prefixes):
        
        from models.resnet_bit import BiT_ResNet, default_bit_resnet_cfg, layers_bit_resnet_cfg 
        Net = BiT_ResNet
        model_cfg = default_bit_resnet_cfg.copy()
        layers_cfg = layers_bit_resnet_cfg

        if model_name in layers_cfg.keys():
            model_cfg['layers']=layers_cfg[model_name]
        else:
            logging.warning(f"Model name {model_name} is not a valid key for layers_cfg. Using default BiT_R50x1")
            model_name = 'BiT_R50x1'

        model_cfg['model_name'] = model_name
    
    elif model_name.startswith(antialiased_resnet_prefixes):
        
        from models.antialiased_resnet import (AntiAliased_ResNet,
                                               default_antialiased_resnet_cfg,
                                               layers_antialiased_resnet_cfg)
        Net = AntiAliased_ResNet
        model_cfg = default_antialiased_resnet_cfg.copy()
        layers_cfg = layers_antialiased_resnet_cfg
        
        if model_name.endswith("_lpf2"):
            model_cfg['filter_size'] = 2
        elif model_name.endswith("_lpf4"):
            model_cfg['filter_size'] = 4
        elif model_name.endswith("_lpf6"):
            model_cfg['filter_size'] = 6

        
        if model_name in layers_cfg.keys():
            model_cfg['layers']=layers_cfg[model_name]
        else:
            logging.warning(f"Model name {model_name} is not a valid key for layers_cfg. Using default resnet18 layer config")
            model_name = 'antialiased_resnet18'

        model_cfg['model_name'] = model_name

    elif model_name.startswith(vit_prefixes):
        
        from models.vit import ViT, default_vit_cfg, layers_vit_cfg
        Net = ViT
        model_cfg = default_vit_cfg.copy()
        layers_cfg = layers_vit_cfg

        if model_name in layers_cfg.keys():
            model_cfg['layers']=layers_cfg[model_name]
        else:
            logging.warning(f"Model name {model_name} is not a valid key for layers_cfg.  Using default vit_tiny layer config")
            model_name = 'vit_tiny'

        model_cfg['model_name'] = model_name

    elif model_name.startswith(mlp_prefixes):
        
        from models.mlp import MLP, default_mlp_cfg, layers_mlp_cfg
        Net = MLP
        model_cfg = default_mlp_cfg.copy()
        layers_cfg = layers_mlp_cfg

        if model_name in layers_cfg.keys():
            model_cfg['layers']=layers_cfg[model_name]
        else:
            logging.warning(f"Model name {model_name} is not a valid key for layers_cfg.  Using default mixer_s16_224 layer config")
            model_name = 'mixer_s16_224'

        model_cfg['model_name'] = model_name


    else:
        logging.error("Error: Model name %s is not currently supported. Exiting." %model_name)
        model_cfg = None

        return model_cfg, model_name

    model_cfg["net_class"] = Net

    if isinstance(args,str):
        if (model_name.startswith(resnet_prefixes) or model_name.startswith(resnet_bit_prefixes)):
            logging.warning("get_model got a string argument. In this case, the corresponding default model_cfg is returned. If relevant, confirm if batchnorm/dropout/drop_path parameters are updated elsewhere.")
        return model_cfg, model_name

    if args.groupnorm:
        model_cfg['groupnorm'] = True
    elif args.batchnorm:
        model_cfg['batchnorm'] = True
    elif args.layernorm:
        model_cfg['layernorm'] = True
    if args.patchify:
        model_cfg['patchify'] = True

    #print(args.dropout,args.drop_path, args.resize)

    if args.dropout is not None: model_cfg['dropout'] = args.dropout
    if args.drop_path is not None: model_cfg['drop_path'] = args.drop_path
    if args.resize is not None and not(model_name.startswith(vit_prefixes)): model_cfg['resize'] = args.resize

    return model_cfg, model_name