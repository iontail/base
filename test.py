import random
import torch
import numpy as np

from arguments import parse_arguments
from data.dataloader import get_dataloader
from models.factory import get_model
from architecture import SL

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


def test():
    args = parse_arguments()

    _setup_reproducibility(42)

    
    test_dl = get_dataloader(name=args.data,
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
    
    
    model_path = f'./checkpoints/{args.model}_best.pth' # change the path
    checkpoint = torch.load(model_path, map_location=device)
    

    model.to(device)
    model.load_state_dict(checkpoint)
    if args.learning == 'sl':
        trainer = SL(model=model, args=args, device=device)
    else:
        pass

    final_loss, final_acc = trainer.evaluate(test_dl)
    print(f"Final Results on Validation: Loss = {final_loss:.4f} | Accuracy = {final_acc:.2f}%")


if __name__ == '__main__':
    test()