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
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False)

        # Sparse Module List
        self.layers = nn.ModuleList()
        self.max_depth = 2**(C-1)
        self.C = C
        self.local_drop_p = local_drop_p
        self.global_drop_ratio = global_drop_ratio

        self.num_layers_row = [0] * self.max_depth # number of layers in each row

        for c in range(C):
            num_layers = 2**(c)
            layer_list = [None] * self.max_depth
            layer_stride = self.max_depth // num_layers
            first_layer_idx = layer_stride - 1
            
            for i in range(first_layer_idx, self.max_depth, layer_stride):
                layer_list[i] = ConvBlock(out_channels, out_channels, drop_p)
                self.num_layers_row[i] += 1

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
 

    def _make_mask(self, row: int, batch: int, g_drop_col: torch.Tensor):
        num_global = g_drop_col.shape[0]
        num_local = batch - num_global

        num_active = self.num_layers_row[row]

        # local drop column masking
        p = torch.full((num_active, num_local), fill_value=(1 - self.local_drop_p)) # (active_column, local)
        local_mask = torch.bernoulli(p)

        # guarantee that at least one path exists in a join operation
        local_sum = local_mask.sum(dim=0)
        local_dead = (local_sum == 0) # (local, )
        if local_dead.any():
            local_dead_idx = local_dead.nonzero(as_tuple=True)[0]
            rand_chosen_col = torch.randint(low=0, high=num_active, size=(local_dead_idx.shape))
            local_mask[rand_chosen_col, local_dead_idx] = 1.0

        # global drop column masking
        # g_drop_col can be [0, self.C]
        # some columns values may not be in num_active_col
        global_mask = torch.zeros(num_active, num_global)
        g_col = g_drop_col - (self.C - num_active) # make index of first active columns 0
        valid = g_col >= 0 # samples that have valid column within global sample
        
        if valid.any():
            valid_mask = valid.nonzero(as_tuple=True)[0]
            valid_global_col = g_col[valid]
            global_mask[valid_global_col, valid_mask] = 1.0

        mask = torch.hstack((global_mask, local_mask)) # (active_column, B)
        return mask
    
    def _join(self, x: torch.Tensor, row: int, g_drop_col: torch.Tensor):
        """
        x: (active_column, B, C, H, W). "active_column" is the number of columns which has a layer in the specific row.
        g_drop_col: (# of global drop,)
        """

        if self.training:
            mask = self._make_mask(row, x.shape[1], g_drop_col).to(x.device) # (active_column, B)
            mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (active_column, B, 1, 1, 1)
            masked_sum = (x * mask).sum(dim=0) # (B, C, H, W)
            num_active = mask.sum(dim=0) # (B,)
            joined = masked_sum / (num_active.reshape(-1, 1, 1, 1) + 1e-6) # (B, C, H, W)

        else:
            num_active_col = self.num_layers_row[row]
            joined = x.sum(dim=0) / (num_active_col + 1e-6)
        
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

            if (self.training) or (col is None):
                # based on 발견1
                layer_start_idx_row = self.C - self.num_layers_row[i]
                last_col = self.C
            else: # inference for the specific column
                layer_start_idx_row = col
                last_col = col + 1
            
            current = []
            for j in range(layer_start_idx_row, last_col):
                layer = self.layers[j][i]
                if layer is not None:
                    current.append(layer(outputs[j]))

            if current:
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
        self.pool_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(self.B-1)])
        self.pool_layers.append(nn.AdaptiveAvgPool2d(1))

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, col: int = None):
        out = self.stem(x)

        g_drop_col = None
        if self.training:
            num_global = int(self.global_drop_ratio * x.shape[0])
            g_drop_col = torch.randint(low=0, high=self.C, size=(num_global,)) # [0, self.C - 1]
     
        for i, block in enumerate(self.blocks):
            out = block(out, g_drop_col, col)

            out = self.pool_layers[i](out)

        out = self.classifier(out)
        return out



def get_fractalnet(model_name: str,
                   num_classes: int
                   ):

    model_config_dict = {
        'fractalnet': ([128, 256, 512, 1024], 4, 4),
        'cifarfractalnet':  ([64, 128, 256, 512, 512], 5, 4)
        }
    
    if model_name not in model_config_dict.keys():
        raise ValueError(f"Given model name does not exit in FractalNet model config. Got {model_name}")

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
    model = get_fractalnet('fractalnet', 1000).to(device) # for comparison with # of params in table 6

    model_summary(model)

    
    with torch.no_grad():
        model.eval()

        data = torch.randn(2, 3, 224, 224).to(device)
        output = model(data)
        pred = output.argmax(dim=-1)

        print(f"Output shape: {output.shape}")
        print(f"Predictions: {pred}")

        output = model(data, col=2)
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
        

