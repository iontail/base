import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms

from .trainer import Trainer, KNN


class NXTentLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()

        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        batch = z_i.size(0)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0) # (2B, D)

        sim = torch.matmul(z, z.T) / self.temperature # (2B, 2B)
        mask = torch.eye(batch * 2, dtype=torch.bool, device = z.device)
        logits = sim.masked_fill(mask, float('-inf'))

        labels = torch.arange(batch, device=z.device)
        labels = torch.cat([labels + batch, labels], dim=0) # 2B

        # n-th image must match (n + batch)-th image
        # conversely, (n + batch)-th image must match n-th image
        loss = self.ce(logits, labels)
        return logits, loss


class SimCLR(Trainer):
    def __init__(self,
                 model: nn.Module,
                 args,
                 device: str = 'cpu'
                 ):


        self.model = model
        # change classifier layer as projection head for implementatione simplicity
        hidden_channels = self.model.feature_dim
        self.model.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.model.to(device)
        super().__init__(model, args, device) # assign params of the modified model optimizer

        self.criterion = NXTentLoss(args.temperature)
        self.knn = KNN()

        if args.data == 'cifar10':
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        elif args.data == 'cifar100':
            normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.5071, 0.4865, 0.4409])

        self.transforms_view1 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            normalize
        ])

        self.transforms_view2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            normalize
            ])
         
    def _forward_epoch(self, loader, get_acc:bool = False):
        samples = 0
        total_loss = 0

        penultimate_features_list = []
        targets_list = []
        

        if self.model.training:
            for batch in tqdm(loader, leave=False):

                data= batch['data']
                targets = batch['targets'].to(self.device)
                samples += data.size(0)

                x_i = torch.stack([self.transforms_view1(img) for img in data], dim=0).to(self.device)
                x_j = torch.stack([self.transforms_view2(img) for img in data], dim=0).to(self.device)

                z_i, h_i = self.model(x_i, penultimate=True)
                z_j, _ = self.model(x_j, penultimate=True)
                penultimate_features_list.append(h_i.detach())
                targets_list.append(targets)

                _, loss = self.criterion(z_i, z_j)
                total_loss += loss.item() * (data.size(0) * 2)
                
                    
                if self.model.training:
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
                    self.optimizer.step()

            if get_acc:
                penultimate_features = torch.cat(penultimate_features_list, dim=0)
                targets = torch.cat(targets_list, dim=0)
                self.knn.fit(penultimate_features, targets)
                
            avg_loss = total_loss / (samples * 2)
            acc = 0

        else:
            if get_acc:
                correct = 0
                samples = 0
                with torch.no_grad():
                    for batch in loader:
                        data= batch['data']
                        targets = batch['targets'].to(self.device)

                        x_i = torch.stack([self.transforms_view1(img) for img in data], dim=0).to(self.device)
                        z_i, h_i = self.model(x_i, penultimate=True)
                        pred = self.knn.predict(h_i, metric='euclidean')
                        correct += (pred == targets).sum().item()
                        samples += data.size(0)
                    acc = correct / samples * 100
                avg_loss = 0
            else:
                acc = 0
                avg_loss = 0     
        return (avg_loss, acc)
    

    def evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():
            loss, acc = self._forward_epoch(loader, get_acc=True)
        return loss, acc
    
    def _compute_loss_correct(self, data: torch.Tensor, targets: torch.Tensor, **kwargs):
        pass