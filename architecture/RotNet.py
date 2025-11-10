import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import Trainer

class RotNet(Trainer):
    def __init__(self,
                 model: nn.Module,
                 args,
                 device: str = 'cpu'
                 ):

        super().__init__(model, args, device)
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def _compute_loss_correct(self, data: torch.Tensor, targets: torch.Tensor, **kwargs):
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        return loss
    
    def _forward_epoch(self, loader, get_acc:bool = False):
        samples = 0
        total_loss = 0

        penultimate_features_list = []
        targets_list = []
        

        for batch in tqdm(loader, leave=False):

            data= batch['data'].to(self.device)
            targets = batch['targets'].to(self.device)

            rotated_data_list = []
            rotated_targets_list = []

            # rotate the images and make new targets(rotation)
            for k in range(4):
                rotated_batch = torch.rot90(data, k=k, dims=[2, 3])
                rotated_data_list.append(rotated_batch)
                rotated_targets_list.append(torch.full((data.size(0),), k, device=data.device))

            rotated_data = torch.cat(rotated_data_list, dim=0)
            rotated_targets = torch.cat(rotated_targets_list, dim=0)
            samples += data.size(0) * 4

            loss = self._compute_loss_correct(rotated_data, targets=rotated_targets)
            total_loss += loss.item() * rotated_data.size(0)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
                self.optimizer.step()

        if get_acc:
            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    data= batch['data'].to(self.device)
                    targets = batch['targets'].to(self.device)

                    _, penultimate_features_batch = self.model(data, penultimate=True)
                    penultimate_features_list.append(penultimate_features_batch)
                    targets_list.append(targets)

                penultimate_features = torch.cat(penultimate_features_list, dim=0)
                targets = torch.cat(targets_list, dim=0)

                acc = self._compute_knn(penultimate_features, targets, metric='euclidean')
        else:
            acc = None
        
        avg_loss = total_loss / samples
        return (avg_loss, acc)
    
    def _compute_knn(self, penultimate_features, targets_list, metric:str='euclidean'):
        """
        Compute knn metric by loop
        Args:
            penultimate_features: (N, D) tensor
            targets_list: (N,) tensor
        """
        # compute knn metric by loop
        N = penultimate_features.size(0)
        correct = 0

        for i in range(N):

            feature_i = penultimate_features[i]

            if metric == 'euclidean':
                distances = torch.pow(feature_i - penultimate_features, 2).sum(dim=1)
            elif metric == 'cosine':
                # cosine sim's input should be 2D tensor (1, D)
                distances = 1 - torch.nn.functional.cosine_similarity(feature_i.unsqueeze(0), penultimate_features, dim=1)

            distances[i] = float('inf')
            nn_id = torch.argmin(distances)

            predicted_label = targets_list[nn_id]

            if predicted_label == targets_list[i]:
                correct += 1


        acc = correct / N * 100
        return acc

    def evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():
            loss, acc = self._forward_epoch(loader, get_acc=True)
        return loss, acc