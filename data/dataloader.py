import torch
from torch.utils.data import DataLoader

from .dataset import get_dataset


def basic_collate_fn(batch):
    data_list = [data[0] for data in batch]
    target_list = [data[1] for data in batch]

    data = torch.stack(data_list)
    targets = torch.tensor(target_list)

    return {
        'data': data,
        'targets': targets
    }

def get_dataloader(name: str = 'cifar10', root: str = './data', train: bool = False, args = None):
    

    dataset = get_dataset(
        name=name,
        root=root,
        train=train,
        default_augment=args.default_augment,
        rand_augment=args.rand_augment,
        color_jitter=args.color_jitter,
        rand_erasing=args.rand_erasing
    )

    shuffle = True if train else False
    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      shuffle=shuffle,
                      num_workers=4,
                      collate_fn=basic_collate_fn
                      )



if __name__ == '__main__':
    pass
