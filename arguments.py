import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()


    # model
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--device', type=str, default='auto')

    # model specified arguments
    parser.add_argument('--growth_rate', type=int, default=12,
                        help="Growth rate for DenseNet100")

    # data
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--default_augment', action='store_true')
    parser.add_argument('--rand_augment', action='store_true')
    parser.add_argument('--color_jittering', action='store_true')
    parser.add_argument('--rand_erasing', action='store_true')

    # training
    parser.add_argument('--learning', type=str, default='sl')
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--scheduler', type=str, default='constant')
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=-1.0) # if <0, deactivate gradient clip

    # log
    parser.add_argument('--use_wandb', action='store_false')
    parser.add_argument('--val_freq', type=int, default=10)

    args = parser.parse_args()
    return args