import torch
import torch.nn as nn
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


        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = self._setup_optimizer(args.optimizer)
        self.use_grad_clip = False
        self.grad_clip = 1.0

        self.milestones = None # if you specify milestone, then define this instance variable
        self.scheduler = get_scheduler(self.optimizer,
                                       args.scheduler,
                                       args.warmup_epochs,
                                       0,
                                       args.epochs,
                                       1e-6,
                                       self.milestones
                                       )
        
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            wandb.init(
                project=f"ELLab_basecode",
                name=f"{args.model.lower()}",
                config={
                    'model': args.model,
                    'data': args.data,
                    'optimizer': args.optimizer,
                    'scheduler': args.scheduler,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'cutmix': args.cutmix,
                    'weight_decay': args.weight_decay
                }
            )

        self.eval_freq = args.eval_freq

        
    def train(self, train_DL, val_DL):
        for epoch in range(self.epochs):
            
            self.model.train()
            loss, acc = self._forward_epoch(train_DL, epoch)

            train_metrics = {'loss': loss, 'acc': acc}


            if (epoch + 1) % self.eval_freq == 0:
                
                self.model.eval()
                with torch.no_grad():
                    val_loss, val_acc = self._forward_epoch(val_DL, epoch)
                    val_metrics = {'loss': val_loss, 'acc': val_acc}

            else:
                val_metrics = {}

            all_metrics = {'train': train_metrics, 'val': val_metrics}
            self._log_metrics(all_metrics, epoch)


    def _log_metrics(self, metrics, epoch):

        log_list = []
        for phase, results in metrics.items():
            if not results:
                continue

            if self.use_wandb:
                metric_dict = {f"{phase}/{k}": v for k, v in results.items()}
                wandb.log(metric_dict, step=epoch)

            metric_items = [f"{k}: {v:.4f}" for k, v in results.items()]
            log_list.append(f"{phase.upper()}: {' | '.join(metric_items)}")


        print(f"Epoch {epoch} | {' | '.join(log_list)}")

        current_lr = self.optimizer.param_groups[0]['lr']
        if self.wandb_run:
            wandb.log({'learning_rate': current_lr}, step=epoch)

        
            


            







    def _forward_epoch(self, loader: DataLoader):
        
        samples = 0
        correct = 0
        total_loss = 0
        
        for batch in tqdm(loader, leave=False):

            data= batch['data'].to(self.device)
            targets = batch['target'].to(self.devie)
            samples += data.size(0)

            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()

            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            preds = outputs.argmax(outputs, dim=-1)
            correct += (preds == targets).sum().item()

        acc = correct / samples * 100
        return (total_loss / samples, acc)
    


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

        

            


