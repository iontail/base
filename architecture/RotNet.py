import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import Trainer, KNN

class RotNet(Trainer):
    def __init__(self,
                 model: nn.Module,
                 args,
                 device: str = 'cpu'
                 ):

        super().__init__(model, args, device)
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.knn = KNN()

    def _compute_loss_correct(self, data: torch.Tensor, targets: torch.Tensor, **kwargs):
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        return loss
    
    def _forward_epoch(self, loader, get_acc:bool = False):
        samples = 0
        total_loss = 0

        penultimate_features_list = []
        targets_list = []
        

        if self.model.training:
            for batch in tqdm(loader, leave=False):

                data= batch['data'].to(self.device)
                targets = batch['targets'].to(self.device)

                rotated_data_list = []
                rotated_targets_list = []

                # rotate the images and make new targets(rotation)
                for k in range(4):
                    rotated_batch = torch.rot90(data, k=k, dims=[2, 3])
                    rotated_data_list.append(rotated_batch)
                    rotated_targets_list.append(torch.full((data.size(0),), k, device=data.device, dtype=torch.long))

                rotated_data = torch.cat(rotated_data_list, dim=0)
                rotated_targets = torch.cat(rotated_targets_list, dim=0)
                samples += data.size(0) * 4

                if get_acc:
                    outputs, penultimate_features_batch = self.model(rotated_data, penultimate=True)
                    penultimate_features_list.append(penultimate_features_batch[:data.size(0)].detach())
                    targets_list.append(targets)
                else:
                    outputs = self.model(rotated_data)

                loss = self.criterion(outputs, rotated_targets)
                total_loss += loss.item() * rotated_data.size(0)
                    

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
                
            avg_loss = total_loss / samples
            acc = 0

        else:
            if get_acc:
                correct = 0
                samples = 0
                with torch.no_grad():
                    for batch in loader:
                        data= batch['data'].to(self.device)
                        targets = batch['targets'].to(self.device)

                        _, penultimate_features_batch = self.model(data, penultimate=True)
                        pred = self.knn.predict(penultimate_features_batch, metric='euclidean')
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