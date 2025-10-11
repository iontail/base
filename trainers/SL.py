import torch
import torch.nn as nn
import tqdm

class SL(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 device: str,
                 use_grad_clip: bool,
                 grad_clip: float
                 ):
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_grad_clip = use_grad_clip
        self.grad_clip = grad_clip

    def forward(self, loader):
        samples = 0
        correct = 0
        total_loss = 0
        
        for batch in tqdm(loader, leave=False):

            data= batch['data'].to(self.device)
            targets = batch['targets'].to(self.device)
            samples += data.size(0)

            outputs, loss = self.model(data, targets)
            total_loss += loss.item() * data.size(0)

            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
                self.optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()

        acc = correct / samples * 100
        avg_loss = total_loss / samples
        return (avg_loss, acc)
