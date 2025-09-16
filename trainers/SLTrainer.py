import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW

from .scheduler import get_scheduler

class SLTrainer:
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
        

    def train(self, loader):
        for epoch in range(self.epochs):
            loss, acc = self._forward_epoch(loader, epoch)

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

        

            


