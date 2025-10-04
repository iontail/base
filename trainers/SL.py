import torch
import torch.nn as nn

class SL(nn.Module):
    def __init__(self, model: nn.Module):

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, targets: torch.Tensor):
        outputs = self.model(x)
        loss = self.criterion(outputs, targets)
        return outputs, loss
