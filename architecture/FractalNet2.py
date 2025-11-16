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


class FractalBlock2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 C: int,
                 depth: int,
                 local_drop_p: float = 0.15,
                 drop_p: float = 0.0
                 ):
        
        super().__init__()

        self.C = C
        self.depth = depth
        self.local_drop_p = local_drop_p

        self.downsample = None
        if in_channels != out_channels and depth == C:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False)
            conv_in_channels = out_channels
        else:
            conv_in_channels = in_channels


        self.current_layer = ConvBlock(in_channels=conv_in_channels,
                                       out_channels=out_channels,
                                       drop_p=drop_p
                                       )
        if depth == 1:
            self.sub_layers = None
        else: # recursively create fractal blocks
            self.sub_layers = nn.ModuleList(
                [FractalBlock2(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    C=C,
                    depth=depth-1,
                    local_drop_p=local_drop_p,
                    drop_p=drop_p
                ) for _ in range(2)]
            )

    def _make_mask(self, g_drop_col, batch, device):
        num_global = g_drop_col.shape[0]
        num_local = batch - num_global

        # local mask
        p = torch.full((2, num_local), fill_value=(1 - self.local_drop_p), device=device)
        local_mask = torch.bernoulli(p)

        # guarantee that at least one path exists in a join operation
        local_sum = local_mask.sum(dim=0)
        local_dead = (local_sum == 0) # (num_local,)
        if local_dead.any():
            local_dead_idx = local_dead.nonzero(as_tuple=True)[0]
            rand_path = torch.randint(0, 2, (local_dead_idx.shape[0],), device=device)
            local_mask[rand_path, local_dead_idx] = 1.0

        

        # global mask
        col_index = self.depth - 1
        global_mask = torch.zeros(2, num_global, device=device) 
        
        # case 1: g_drop_col == col_index -> not consider sublayer's output
        is_current_col = (g_drop_col == col_index)
        global_mask[0, is_current_col] = 1.0 # current_out
        global_mask[1, is_current_col] = 0.0 # sub_out

        # case 2: g_drop_col < col_index
        is_sub_col = (g_drop_col < col_index)
        global_mask[0, is_sub_col] = 0.0 # current_out
        global_mask[1, is_sub_col] = 1.0 # sub_out

        mask = torch.hstack((global_mask, local_mask)) # (2, B)
        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (2, B, 1, 1, 1)
        return mask

    def _join(self, outs: list[torch.Tensor], g_drop_col: torch.Tensor):
        stacked_outs = torch.stack(outs, dim=0)  # (2, B, C, H, W)
        B = stacked_outs.size(1)

        if self.training:
            mask = self._make_mask(g_drop_col, B, stacked_outs.device) # (2, B, 1, 1, 1)      
            masked_sum = (stacked_outs * mask).sum(dim=0) # (B, C, H, W)
            num_active_paths = mask.sum(dim=0) # (B, 1, 1, 1)
            
            joined = masked_sum / (num_active_paths + 1e-6)

        else:
            joined = stacked_outs.mean(dim=0)
        return joined

    def forward(self, x: torch.Tensor, g_drop_col: torch.Tensor):
        if self.downsample is not None:
            x = self.downsample(x)

        current_out = self.current_layer(x)

        if self.sub_layers is None: # end of recursion
            return current_out
        
        sub_out = self.sub_layers[1](self.sub_layers[0](x, g_drop_col), g_drop_col)
        out = self._join([current_out, sub_out], g_drop_col)
        return out
    

class FractalNet2(nn.Module):
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
            block = FractalBlock2(
                in_channels=current_channels,
                out_channels=channel_list[i],
                C=C,
                depth=C,
                local_drop_p=local_drop_p,
                drop_p=drop_p_list[i]
            )

            self.blocks.append(block)
            current_channels = channel_list[i]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Linear(current_channels, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)


    @property
    def feature_dim(self):
        return self.classifier.in_features
    
    def forward(self, x: torch.Tensor, penultimate: bool = False):
        x = self.stem(x)

        if self.training:
            num_global = int(self.global_drop_ratio * x.shape[0])
            g_drop_col = torch.randint(low=0, high=self.C, size=(num_global,), device=x.device)
        else:
            g_drop_col = torch.empty(0, dtype=torch.long, device=x.device)
     
        for i, block in enumerate(self.blocks):
            x = block(x, g_drop_col)
            x = self.pool_layers[i](x)

        x = self.head(x)
        out = self.classifier(x)
        if penultimate:
            return out, x
        else:
            return out



def get_fractalnet2(model_name: str,
                   num_classes: int
                   ):


    model_config_dict = {
        'fractalnet': ([128, 256, 512, 1024], 4, 4),
        'cifarfractalnet':  ([64, 128, 256, 512, 512], 5, 4)
        }
    
    if model_name not in model_config_dict.keys():
        raise ValueError(f"Given model name does not exit in FractalNet model config. Got {model_name}")

    model_config = model_config_dict[model_name]
    return FractalNet2(
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
    model = get_fractalnet2('fractalnet', 1000).to(device) # for comparison with # of params in table 6

    model_summary(model)

    
    with torch.no_grad():
        model.eval()

        data = torch.randn(2, 3, 224, 224).to(device)
        output = model(data)
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
        