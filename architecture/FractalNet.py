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
            nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()
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

        num_layers_row = [0] * self.max_depth # number of layers in each row

        for c in range(C):
            num_layers = 2**(c)
            layer_list = [None] * self.max_depth
            layer_stride = self.max_depth // num_layers
            fist_layer_idx = layer_stride - 1
            
            for i in range(layer_stride - 1, self.max_depth, layer_stride):
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

        self.num_layers_row = num_layers_row
 

    def _make_mask(self, input_shape: list, row: int, g_drop_col: torch.Tensor):
        B = input_shape[1] # batch
        num_global = g_drop_col.shape[0]
        num_local = self.C - num_global

        # local drop column masking
        p = torch.full_like([self.C, B], fill_value=self.local_drop_p) # (column(=self.C), B)
        mask = torch.bernoulli(p)

        # global drop column masking
        # g_drop_col can be [0, self.C]
        mask[g_drop_col, :num_global] = 1.0

        # make the columns zero in the row which have no layers
        num_layer_row = self.num_layers_row[row]
        non_active_col = self.C - num_layer_row
        mask[:non_active_col, num_global:] = 0.0

        # guarantee that at least one path exists in a join operation
        active_sum = mask.sum(dim=0) # (B,)
        if active_sum[num_global:].any():
            make_active_idx = torch.where(active_sum == 0, 1.0, 0.0)
            rand_chosen_col = torch.randint(low=non_active_col, high=self.C, size=[make_active_idx.shape])
            mask[rand_chosen_col[non_active_col:]] = 1.0

        return mask
    
    def _join(self, x: torch.Tensor, row: int, g_drop_col: torch.Tensor):
        """
        x: (# of columns, B, C, H, W). # of columns is the number of columns which has a layer in the specific row.
        g_drop_col: (# of global drop,)
        """

        if self.training:
            mask = self._make_mask(x.shape, row, g_drop_col) # (# of columns, B)
            mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (# of columns, B, 1, 1, 1)
            masked_sum = x[mask].sum(dim=0) # (B,)
            num_active = mask.sum(dim=0) # (B,)
            joined = masked_sum / num_active

        else:
            num_layer_row = self.num_layers_row[row]
            joined = x.sum(dim=0) / num_layer_row
        
        return joined

    def forward(self, x:  torch.Tensor, g_drop_col: torch.Tensor, col: int = None):
        """
        x: (B, C, H, W)
        g_drop_col: (# of global drop,)
        col: which column to use in inference
        """

        if self.downsample is not None:
            x = self.downsample(x)

        outputs = [x] * self.C
        for i in range(self.max_depth):

            if self.training:
                # based on 발견1
                layer_start_idx_row = self.C - self.num_layers_row[i]
                last_col = self.C
            elif col is None:
                layer_start_idx_row = self.C - 1
                last_col = self.C
            else: # inference for the specific column
                layer_start_idx_row = col
                last_col = col + 1
            
            current = []
            for j in range(layer_start_idx_row, last_col):
                layer = self.layers[i][j]
                current.append(layer(outputs[j]))

            stacked_current = torch.stack(current, dim=0) # (# of columns, B, C, H, W)
            row_out = self._join(stacked_current, row=i, g_drop_col=g_drop_col)

            for j in range(layer_start_idx_row, self.C):
                outputs[j] = row_out

        # output has same result in each index
        return outputs[0]

            
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

        self.B = B
        self.C = C
        self.local_drop_p = local_drop_p
        self.global_drop_ratio = global_drop_ratio
    
        if drop_p:
            drop_p_list = [0.1 * i for i in range(B)] # [0.0, 0.1, 0.2, 0.3, 0.4]
        else:
            drop_p_list = [0.0] * B

        current_channels = channel_list[0]
        self.stem = ConvBlock(3, current_channels, drop_p = 0.0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes)
        )

    def forward(self, x: torch.Tensor, col: int = None):
        out = self.stem(x)

        g_drop_col = None
        if self.training:
            num_global = int(self.global_drop_ratio * x.shape[0])
            g_drop_col = torch.randint(low=0, high=self.C, size=(num_global)) # [0, self.C - 1]
     
        for i, block in enumerate(self.blocks):
            out = block(out, g_drop_col, col)

            if i != (self.B - 1):
                out = self.maxpool(out)

        out = self.classifier(out)



def get_fractalnet(model_name: str,
                   num_classes: int
                   ):

    model_config_dict = {
        'fractalnet': ([128, 256, 512, 1024], 5, 4),
        'cifarfractalnet':  ([64, 128, 256, 512, 512], 5, 4)
        }

    # special case for DenseNet100 k=12 or k=24
    if model_name not in model_config_dict.keys():
        raise ValueError(f"Given model name does not exit in ViT model config. Got {model_name}")

    model_config = model_config_dict[model_name]
    return FractalNet(
        num_classes=num_classes,
        channel_list=model_config[0],
        B=model_config[1],
        C=model_config[2],
        local_drop_p=0.15,
        global_drop_ratio=0.5,
        drop_p=0.0
    )


if __name__ == '__main__':
    import torch

    def model_summary(model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p  in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


    device = 'cpu'
    model = get_fractalnet('cifarfractal', 10).to(device) # for comparison with # of params in table 6

    model_summary(model)

    
    with torch.no_grad():
        model.eval()

        data = torch.randn(2, 3, 32, 32).to(device)
        output = model(data)
        pred = output.argmax(dim=-1)

        print(f"Output shape: {output.shape}")
        print(f"Predictions: {pred}")

        output = model(data, col=3)
        pred = output.argmax(dim=-1)

        print(f"Output shape: {output.shape}")
        print(f"Predictions: {pred}")

    model.train()

    output = model(data)
    pred = output.argmax(dim=-1)

    print(f"Output shape: {output.shape}")
    print(f"Predictions: {pred}")
    ###################################
    # Complete Model Checking
        

