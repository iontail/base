from .ResNet import get_resnet

def get_model(model: str, num_classes: int, is_data_small: bool = True):
    
    if 'resnet' in model:
        model = get_resnet(model_name=model, num_classes=num_classes, is_data_small=is_data_small)


    return get_resnet