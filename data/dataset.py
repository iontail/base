from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms


def get_dataset(name: str = 'cifar10',
                root: str = './data',
                train: bool = False,
                default_augment: bool = True,
                rand_augment: bool = False,
                color_jitter: bool = False,
                rand_erasing: bool = False
                ):
    
    name = name.lower()
    
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2470, 0.2435, 0.2616]

    cifar100_mean = [0.5071, 0.4865, 0.4409]
    cifar100_std = [0.2673, 0.2564, 0.2762]


    default_augment = [transforms.ToTensor()]
    if name == 'cifar10':
        default_augment.append(transforms.Normalize(mean=cifar10_mean, std=cifar10_std))
    elif name == 'cifar100':
        default_augment.append(transforms.Normalize(mean=cifar100_mean, std=cifar100_std))

    transform_list = []
    tensor_transform_list = []

    if default_augment:
        transform_list.append(transforms.RandomCrop(32))
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
        
    if rand_augment:
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    if color_jitter:
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    if rand_erasing:
        tensor_transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)))

    total_transforms =transforms.Compose(transform_list + default_augment + tensor_transform_list)


    if name == 'cifar10':
        return CIFAR10(root, train=train, transform=total_transforms, download=True)
    elif name == 'cifar100':
        return CIFAR100(root, train=train, transform=total_transforms, download=True)
    

