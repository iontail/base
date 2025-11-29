# base


```bash
pip install tqdm wandb
apt update && apt install -y tmux
wandb login


python train.py --no_augment --learning=simclr --epochs=100 --warmup_epochs=5 --batch_size=2048 --lr=1e-1 --grad_clip=1.0 --val_freq=1
python train.py --no_augment --learning=moco --epochs=100 --warmup_epochs=5 --batch_size=2048 --lr=1e-1 --grad_clip=1.0 --val_freq=1 --queue_size=8192
```