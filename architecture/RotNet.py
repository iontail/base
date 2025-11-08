import torch
import torch.nn as nn
from tqdm import tqdm
from .trainers import Trainer

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
        # TODO: rotate the input images and adjust the targets accordingly
        outputs, penultimate_features = self.model(data, penultimate=True)
        loss = self.criterion(outputs, targets)
        return loss, penultimate_features
    
    def _forward_epoch(self, loader):
        samples = 0
        total_loss = 0

        penultimate_features = torch.empty((0, self.model.feature_dim)).to(self.device)
        targets_list = torch.empty((0,)).to(self.device)

        for batch in tqdm(loader, leave=False):

            data= batch['data'].to(self.device)
            targets = batch['targets'].to(self.device)
            samples += data.size(0)

            loss, penultimate_output = self._compute_loss_correct(data, targets)
            total_loss = loss.item() * data.size(0)
            penultimate_features = torch.cat((penultimate_features, penultimate_output), dim=0)
            targets_list = torch.cat((targets_list, targets), dim=0)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
                self.optimizer.step()

        acc = self._compute_knn(penultimate_features, targets_list, metric='euclidean')
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
