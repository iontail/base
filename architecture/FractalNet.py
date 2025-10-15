import torch.nn as nn
import torch

class ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 drop_p: float = 0.0
                 ):
        
        self.layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(drop_p) if drop_p > 0 else nn.Identity
        ]

        super().__init__(*self.layers)


class FractalBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 C: int,
                 local_drop_p: float = 0.15,
                 global_drop_ratio: float = 0.5,
                 drop_p: float = 0.0
                 ):
        
        super().__init__()

        self.downsample = None
        if in_channels < out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False)

        # Sparse Module List
        self.layers = nn.ModuleList()
        self.max_depth = 2**(C-1)
        for c in C:
            num_layers = 2**(c)
            layer_list = [None] * self.max_depth
            layer_stride = self.max_depth // num_layers
            fist_layer_idx = layer_stride - 1
            
            for i in range(layer_stride - 1, num_layers, layer_stride):
                if layer_list[fist_layer_idx] == None and self.downsample is None:
                    layer_list[i] = ConvBlock(in_channels, out_channels, drop_p)
                else:
                    layer_list[i] = ConvBlock(out_channels, out_channels, drop_p)

            self.layers.append(nn.ModuleList(layer_list))

        """
        -------------------------------------
        Overview of self.layers in C = 3 case
        -------------------------------------
        None | None | L1
        None | L1   | L2
        None | None | L3
        L1   | L2   | L4
        """

    def global_drop(self, x: torch.Tensor):
        # x: (B, C, H, W)
        pass

    def local_global_drop(self, x: torch.Tensor):
        pass


    def forward(self, x:  torch.Tensor, column: int = None, trainig: bool = False):
        pass

        

class FractalNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 channel_list: list,
                 B: int = 5,
                 C: int = 4,
                 local_drop_p: float = 0.15,
                 global_drop_ratio: float = 0.5,
                 drop_p: bool = False
                 ):
        
        super().__init__()

        if drop_p:
            drop_p_list = [0.1 * i for i in range(B)] # [0.0, 0.1, 0.2, 0.3, 0.4]
        else:
            drop_p_list = [0.0] * B


        current_channels = channel_list[0]
        self.stem = ConvBlock(3, current_channels, drop_p = 0.0)

        self.blocks = nn.ModuleList()

        for i in range(B):
            block = FractalBlock(
                in_channels=current_channels,
                out_channels=channel_list[i],
                C=C,
                local_drop_p=local_drop_p,
                global_drop_ratio=global_drop_ratio,
                drop_p=drop_p_list[i]
            )

            self.blocks.append(block)
            current_channels = channel_list[i]

        self.classifier = nn.Sequential(
            None
        )




        

