from .ResNet import get_resnet
from .PreActResNet import get_preactresnet
from .DenseNet import get_densenet

def get_model(model: str, num_classes: int, is_data_small: bool = True, growth_rate: int = 12):
    
    model = model.lower()
    if 'preactresnet' in model:
        model = get_preactresnet(model_name=model, num_classes=num_classes, is_data_small=is_data_small)

    elif 'resnet' in model:
        model = get_resnet(model_name=model, num_classes=num_classes, is_data_small=is_data_small)

    elif 'densenet' in model:
        model = get_densenet(model_name=model, num_classes=num_classes, growth_rate=growth_rate)


    return model