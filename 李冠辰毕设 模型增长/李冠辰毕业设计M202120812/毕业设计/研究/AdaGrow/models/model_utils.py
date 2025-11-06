import re


MODEL_INFO = {
    "ViT": {
        "saliency_backbone": r"module\.transformer\.layers\.(\d+)\.0\.norm",
    },
    "BertForClassification": {
        "saliency_backbone": r"module\.layers\.(\d+)\.norm2",
    },
    "VGGNet3Block": {
        "saliency_stem": "module.bn1",
        "saliency_backbone": r"module\.downstream(\d+)\.norm",
    }, 
    "VGGNet4Block": {
        "saliency_stem": "module.bn1",
        "saliency_backbone": r"module\.downstream(\d+)\.norm",
    },
    "ResNet3Block": {
        "saliency_stem": "module.bn1",
        "saliency_backbone": r"module\.downstream(\d+)\.norm",
    }, 
    "ResNet4Block": {
        "saliency_stem": "module.bn1",
        "saliency_backbone": r"module\.downstream(\d+)\.norm",
    },
    "GrowingMobileNet3Block": {
        "saliency_stem": "module.bn1",
        "saliency_backbone": r"module\.downstream(\d+)\.norm",
    }, 
    "GrowingMobileNet4Block": {
        "saliency_stem": "module.bn1",
        "saliency_backbone": r"module\.downstream(\d+)\.norm",
    }
}


MODEL_INFO_1GPU = {

    "VGGNet3Block": {
        "saliency_stem": "bn1",
        "saliency_backbone": r"downstream(\d+)\.norm",
    }, 
    "VGGNet4Block": {
        "saliency_stem": "bn1",
        "saliency_backbone": r"downstream(\d+)\.norm",
    },

    "ResNet3Block": {
        "saliency_stem": "bn1",
        "saliency_backbone": r"downstream(\d+)\.norm",
    }, 
    "ResNet4Block": {
        "saliency_stem": "bn1",
        "saliency_backbone": r"downstream(\d+)\.norm",
    },

    "GrowingMobileNet3Block": {
        "saliency_stem": "bn1",
        "saliency_backbone": r"downstream(\d+)\.norm",
    }, 
    "GrowingMobileNet4Block": {
        "saliency_stem": "bn1",
        "saliency_backbone": r"downstream(\d+)\.norm",
    }
}


def get_saliency(model):
    MINFO = MODEL_INFO[model.module.__class__.__name__]
    pattern = re.compile(MINFO["saliency_backbone"])
    
    matched_modules = []
    for name, module in model.named_modules():
        if "saliency_stem" in MINFO.keys() and name == MINFO["saliency_stem"]:
            index = 0
            while len(matched_modules) <= index:
                matched_modules.append(None)
            matched_modules[index] = {
                "name": name,
                "module": module,
                "saliency": module.weight.data.mean().item()
            }
        match = pattern.match(name)
        if match:
            index = int(match.group(1))
            while len(matched_modules) <= index:
                matched_modules.append(None)
            matched_modules[index] = {
                "name": name,
                "module": module,
                "saliency": module.weight.data.mean().item()
            }
    
    return matched_modules


def find_layers(module, layers=["reparameterizer.RepUnit", "RepUnit"], name=''):
    if type(module).__name__ in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def get_inner_layer_saliency(model):
    rep_units = find_layers(model)
    saliency = {}
    for name, module in rep_units.items():
        bn_weights = []
        for _, sub_module in module.torep_extractor.items():
            if type(sub_module).__name__ in ("reparameterizer.RepScaledConv", "RepScaledConv"):
                bn_weight = sub_module.bn.weight.data.mean().item()
                bn_weights.append(bn_weight)
        saliency[name] = sum(bn_weights) / len(bn_weights)
    return saliency


def get_module_by_path(root_module, path):
    attributes = path.split('.')
    module = root_module
    for attr in attributes:
        if attr.isdigit(): 
            module = module[int(attr)]
        else:
            module = getattr(module, attr)
    return module


if __name__ == "__main__":
    from adagrow.ada_growing_vgg import get_ada_growing_vgg
    from adagrow.ada_growing_resnet import get_ada_growing_bottleneck_resnet, get_ada_growing_basic_resnet
    from adagrow.ada_growing_mobilenet import get_ada_growing_mobilenetv3
    model = get_ada_growing_basic_resnet([2,2,2])
    print(get_inner_layer_saliency(model))