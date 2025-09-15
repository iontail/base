import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 stride: int = 1,
                 kernel_size: int = 3
                 ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 0, stride = stride, bias = False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_dim)
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.layers(x)

        x += residual
        x = self.relu(x)
        return x
    

class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 stride: int = 1,
                 kernel_size: int = 3
                 ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = 1, padding = 0, stride = stride, bias = False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding =1, stride = 1, bias = False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, 4 * out_dim, kernel_size = 1, padding =1, stride = 1, bias = False),
            nn.BatchNorm2d(4 * out_dim)
        )

        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor):
        residual = x
        x = self.layers(x)

        x += residual
        x = self.relu(x)
        return x
    

class ResNetBlock(nn.Module):
    pass
    