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
        self.C = C
        self.local_drop_p = local_drop_p
        self.global_drop_ratio = global_drop_ratio

        length = []
        num_layers_row = [0] * C # number of layers in each row

        for c in C:
            num_layers = 2**(c)
            layer_list = [None] * self.max_depth
            layer_stride = self.max_depth // num_layers
            fist_layer_idx = layer_stride - 1
            length.append(num_layers)
            for i in range(layer_stride - 1, num_layers, layer_stride):
                if layer_list[fist_layer_idx] == None and self.downsample is None:
                    layer_list[i] = ConvBlock(in_channels, out_channels, drop_p)
                else:
                    layer_list[i] = ConvBlock(out_channels, out_channels, drop_p)
                num_layers_row[i] += 1

            self.layers.append(nn.ModuleList(layer_list))

        """
        -------------------------------------
        Overview of self.layers in C = 3 case
        -------------------------------------
        None | None | L1
        None | L1   | L2
        None | None | L3
        L1   | L2   | L4

        특징1: 특정 낮은(왼쪽) 열에서 None이 아닌 레이어가 있다면 해당 행에서 그보다 높은(오른쪽) 열에는 반드시 레이어가 존재한다
        발견1: join operation은 특정 행에 존재하는 모든 오른쪽 레이어의 값에 적용하면 된다
        """

        self.length = torch.tensor(length, dtype=torch.int64)
        self.num_layers_row = torch.tensor(num_layers_row, dtype=torch.int64)
 

    def _make_mask(self, x: torch.Tensor):
        pass
    
    def _join(self, x: torch.Tensor):

    def forward(self, x:  torch.Tensor, column: torch.Tensor):
        """
        x: (B, C, H, W)
        column: (# of global drop, column)
        """

        if self.downsample is not None:
            x = self.downsample

        outputs = [x] * self.C
        for i in range(self.max_depth):
            # based on 발견1
            layer_start_idx_row = self.C - self.num_layers_row[i]
            current = []
            for j in range(layer_start_idx_row, self.C):
                layer = self.layers[i][j]
                current.append(layer(outputs[j]))

            row_out = self._join(current)

            for j in range(layer_start_idx_row, self.C):
                outputs[j] = row_out

        return outputs

            

        



        

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




        

