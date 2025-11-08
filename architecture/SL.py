import torch
import torch.nn as nn
from .trainer import Trainer

class SL(Trainer):
    def __init__(self,
                 model: nn.Module,
                 args,
                 device: str = 'cpu'
                 ):

        super().__init__(model, args, device)
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def _compute_loss_correct(self, data: torch.Tensor, targets: torch.Tensor, **kwargs):
        outputs = self.model(data, penultimate=False)
        loss = self.criterion(outputs, targets)

        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return loss, correct
