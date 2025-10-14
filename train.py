import random
import torch
import numpy as np

from arguments import parse_arguments
from data.dataloader import get_dataloader
from architecture import get_model, SL


def _setup_reproducibility(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _setup_device(device: str = 'auto'):

    # I exclude MPS as usually do not use MPS 
    if device == 'auto':
        if torch.cuda.is_available():
            use_device = torch.device('cuda')
        else:
            use_device = torch.device('cpu')

    else:
        if device not in ['cuda', 'cpu']:
            raise ValueError(f"Device must be one of {['cuda', 'cpu']}. Got {device}")
        
        use_device = torch.device(device)
    return use_device


def main():
    args = parse_arguments()

    _setup_reproducibility(42)

    train_dl = get_dataloader(name=args.data,
                              root='./data',
                              train=True,
                              args=args
                              )
    
    val_dl = get_dataloader(name=args.data,
                            root='./data',
                            train=False,
                            args=args
                            )
    
    device = _setup_device(args.device)

    model = get_model(model=args.model,
                      num_classes=args.num_classes,
                      is_data_small=True if args.data in ['cifar10', 'cifar100', 'tinyimagenet'] else False,
                      growth_rate=args.growth_rate,
                      img_size=args.img_size
                      )
    
    model.to(device)
    trainer = Trainer(model=model, args=args, device=device)
    trainer.train(train_dl, val_dl)

    final_loss, final_acc = trainer.evaluate(val_dl)
    print(f"Final Results on Validation: Loss = {final_loss:.4f} | Accuracy = {final_acc:.2f}%")


if __name__ == '__main__':
    main()