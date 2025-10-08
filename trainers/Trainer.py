import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
import wandb

from .scheduler import get_scheduler

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 args,
                 device: str = 'cpu'
                 ):
        
        self.model = model
        self.args = args
        self.device = device
        self.epochs = args.epochs

        self.optimizer = self._setup_optimizer(args.optimizer)
        self.grad_clip = self.args.grad_clip
        self.use_grad_clip = False if self.grad_clip < 0 else True

        self.milestones = [82, 123] # if you specify milestone, then define this instance variable
        #self.milestones = [150, 225]
        self.scheduler = get_scheduler(optimizer=self.optimizer,
                                       scheduler_name=args.scheduler,
                                       warmup_epochs=args.warmup_epochs,
                                       warmup_start_lr=args.warmup_start_lr,
                                       total_epochs=args.epochs,
                                       min_lr=1e-6,
                                       milestones=self.milestones
                                       )
        
        

        self.val_freq = args.val_freq


        os.makedirs('./checkpoints', exist_ok=True)
        prefix = f"{self.args.model.lower()}_best.pth"
        self.save_path = os.path.join('./checkpoints', prefix)
        self.data_name = args.data 

        

        
    def train(self, train_dl, val_dl):

        self.use_wandb = self.args.use_wandb
        if self.use_wandb:
            wandb.init(
                project=f"ELLab_basecode",
                name=f"{self.args.model.lower()}",
                config={
                    'model': self.args.model,
                    'data': self.args.data,
                    'optimizer': self.args.optimizer,
                    'scheduler': self.args.scheduler,
                    'batch_size': self.args.batch_size,
                    'lr': self.args.lr,
                    'cutmix': self.args.cutmix,
                    'weight_decay': self.args.weight_decay
                }
            )

        best = float('-inf')
        for epoch in range(self.epochs):
            
            self.model.train()
            loss, acc = self._forward_epoch(train_dl)

            train_metrics = {'loss': loss, 'acc': acc}

            if ((epoch + 1) % self.val_freq == 0) or (epoch == self.epochs - 1):
                
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_acc = self._forward_epoch(val_dl)
                    val_metrics = {'loss': val_loss, 'acc': val_acc}


                # save checkpoint
                if val_acc > best + 1e-3:
                    best = val_acc
                    torch.save(self.model.state_dict(), self.save_path)

            else:
                val_metrics = {}

            if self.scheduler is not None:
                self.scheduler.step()

            all_metrics = {'train': train_metrics, 'val': val_metrics}
            self._log_metrics(all_metrics, best, epoch)

        if self.use_wandb:
            wandb.finish()


    def _log_metrics(self, metrics, best, epoch):

        log_list = []
        for phase, results in metrics.items():
            if not results:
                continue

            if self.use_wandb:
                metric_dict = {f"{self.data_name}/{phase}/{k}": v for k, v in results.items()}
                wandb.log(metric_dict, step=epoch)

            metric_items = [f"{k}: {v:.4f}" for k, v in results.items()]
            log_list.append(f"{phase}: {' | '.join(metric_items)}")


        print(f"Epoch {epoch} | {' | '.join(log_list)}")

        current_lr = self.optimizer.param_groups[0]['lr']
        if self.use_wandb:
            wandb.log({
                'learning_rate': current_lr,
                'best': best
            }, step=epoch)



    def _forward_epoch(self, loader: DataLoader):
        
        samples = 0
        correct = 0
        total_loss = 0
        
        for batch in tqdm(loader, leave=False):

            data= batch['data'].to(self.device)
            targets = batch['targets'].to(self.device)
            samples += data.size(0)

            outputs, loss = self.model(data, targets)
            total_loss += loss.item() * data.size(0)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
                self.optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()

        acc = correct / samples * 100
        avg_loss = total_loss / samples
        return (avg_loss, acc)
    

    def evaluate(self, loader: DataLoader):

        self.model.eval()
        with torch.no_grad():
            loss, acc = self._forward_epoch(loader)
        return loss, acc
    

    def _setup_optimizer(self, optimizer_name: str = 'sgd'):
        
        optimizer_name = optimizer_name.lower()

        if optimizer_name == 'sgd':
            return SGD(self.model.parameters(),
                       lr = self.args.lr,
                       momentum=0.9,
                       weight_decay=self.args.weight_decay
                       )
        
        elif optimizer_name == 'adam':
            return Adam(self.model.parameters(),
                        lr = self.args.lr,
                        betas=(0.9, 0.999), # uses default (change if needed)
                        weight_decay=self.args.weight_decay
                        )
        
        elif optimizer_name == 'adamw':
            return AdamW(self.model.parameters(),
                        lr = self.args.lr,
                        betas=(0.9, 0.999), # uses default (change if needed)
                        weight_decay=self.args.weight_decay
                        )