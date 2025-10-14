import torch
import torch.nn as nn
from .trainers import Trainer

class SL(Trainer):
    def __init__(self,
                 model: nn.Module,
                 args,
                 device: str = 'cpu'
                 ):

        super().__init__(model, args, device)
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def _get_loss(self, data: torch.Tensor, targets: torch.Tensor):
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        return outputs, loss
