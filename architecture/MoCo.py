import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms

from .trainer import Trainer, KNN


class MoCoLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor):

        sim_pos = (q * k).sum(dim=1, keepdim=True)
        sim_neg = torch.matmul(q, queue)
        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature  # (B, 1+K)

        # labels 0 == positive
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)

        loss = self.ce(logits, labels)
        return logits, loss


# To use nn.Module.register_buffer, I decouple MoCo and MoCo's Trainer
class MoCo(nn.Module):
    def __init__(self, encoder: nn.Module, args, device: str = 'cpu'):
        super(MoCo, self).__init__()

        self.device = device
        self.encoder_q = encoder
        self.momentum = args.moco_momentum

        #freeze encoer_k
        self.encoder_k = copy.deepcopy(encoder)
        self.encoder_k.to(device)
        for params in self.encoder_k.parameters():
            params.requires_grad = False
                                       

        # change classifier layer as projection head for implementatione simplicity
        hidden_channels = self.encoder_q.feature_dim
        self.encoder_q.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.encoder_k.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.queue_size = args.queue_size
        # for reducing computation cost in transposing during computing loss
        # initialize the queue a shape of (hidden_channels, queue_size)
        queue = torch.randn(hidden_channels, self.queue_size)
        queue = nn.functional.normalize(queue, dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))

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
        
        if self.queue_size % args.batch_size != 0:
            raise ValueError("queue_size must be divisible by batch_size")


    @property
    def feature_dim(self):
        return self.encoder_q.feature_dim
    
    @torch.no_grad()
    def _update_encoder(self):
        for params_q, params_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            params_k.data = self.momentum * params_k.data + (1 - self.momentum) * params_q.data


    @torch.no_grad()
    def _update_queue(self, keys: torch.Tensor):
        batch = keys.size(0)
        loc = int(self.queue_pointer)

        # queue: (hidden_channels, queue_size)

        if loc + batch <= self.queue_size:
            self.queue[:, loc:loc + batch] = keys.T
        else:
            non_ramained = self.queue_size - loc
            self.queue[:, loc:] = keys[:non_ramained].T
            self.queue[:, :batch - non_ramained] = keys[non_ramained:].T
            
        self.queue_pointer[0] = (loc + batch) % self.queue_size  # move pointer


    def forward(self, x: torch.Tensor, penultimate: bool = False):
        x_i = torch.stack([self.transforms_view1(img) for img in x], dim=0).to(self.device)
        x_j = torch.stack([self.transforms_view2(img) for img in x], dim=0).to(self.device)

        z_i, h_i = self.encoder_q(x_i, penultimate=True)
        z_i = F.normalize(z_i, dim=1)

        with torch.no_grad():
            self._update_encoder()
            z_j, h_j = self.encoder_k(x_j, penultimate=True)
            z_j = F.normalize(z_j, dim=1)

        if penultimate:
            return z_i, z_j, h_i, h_j
        else:
            return z_i, z_j, None, None

class MoCo_Trainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 args,
                 device: str = 'cpu'
                 ):


        self.model = MoCo(model, args, device)
        super().__init__(self.model, args, device) # assign params of the modified model optimizer
 
        self.criterion = MoCoLoss(args.temperature)
        self.knn = KNN()

             
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

                z_i, z_j, h_i, _= self.model(data, penultimate=True)
                penultimate_features_list.append(h_i.detach())
                targets_list.append(targets.detach())

                _, loss = self.criterion(z_i, z_j, self.model.queue)
                total_loss += loss.item() * data.size(0)
                
                    
                if self.model.training:
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
                    self.optimizer.step()

                self.model._update_queue(z_j)

            if get_acc:
                penultimate_features = torch.cat(penultimate_features_list, dim=0)
                targets = torch.cat(targets_list, dim=0)
                self.knn.fit(penultimate_features, targets)
                
            avg_loss = total_loss / samples
            acc = 0

        else:
            if get_acc:
                correct = 0
                samples = 0
                with torch.no_grad():
                    for batch in loader:
                        data= batch['data']
                        targets = batch['targets'].to(self.device)

                        _, h_i = self.model.encoder_q(data, penultimate=True)
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