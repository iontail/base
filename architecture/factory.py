from .ResNet import get_resnet
from .PreActResNet import get_preactresnet
from .DenseNet import get_densenet
from .ViT import get_vit
from .MLPMixer import get_mlpmixer
from .FractalNet import get_fractalnet

def get_model(model: str, num_classes: int, is_data_small: bool = True, growth_rate: int = 12, img_size: int = 32):
    
    model = model.lower()
    if 'preactresnet' in model:
        model = get_preactresnet(model_name=model, num_classes=num_classes, is_data_small=is_data_small)

    elif 'resnet' in model:
        model = get_resnet(model_name=model, num_classes=num_classes, is_data_small=is_data_small)

    elif 'densenet' in model:
        model = get_densenet(model_name=model, num_classes=num_classes, growth_rate=growth_rate)

    elif 'vit' in model:
        model = get_vit(model_name=model, num_classes=num_classes, img_size=img_size)

    elif 'mlpmixer' in model:
        model = get_mlpmixer(model_name=model, num_classes=num_classes, img_size=img_size)

    elif 'fractalnet' in model:
        model = get_fractalnet(model_name=model, num_classes=num_classes)

    return model