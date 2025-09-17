import torch.nn as nn
import torch


# the original paper use 'block' term instead of stage or convN
# so I used 'layer' term which stands for 'block' in ResNet code
class DenseLayer(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 growth_rate: int,
                 bottleneck: bool = False
                 ):
        
        if bottleneck:
            layers = [
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(4 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, stride=1, bias=False), 
            ]

        else:
            layers = [
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, stride=1, bias=False)
            ]

        super().__init__(*layers) # forward method is automatically 


class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_blocks: int,
                 growth_rate: int,
                 bottleneck: bool = False
                 ):
        
        super().__init__()

        layers = []
        for i in range(num_blocks):
            changed_in_channels = in_channels + i * growth_rate
            layers.append(DenseLayer(changed_in_channels, growth_rate, bottleneck))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        concatenated_x = [x]

        for layer in self.layers:
            x = layer(torch.cat(concatenated_x, dim=1))
            concatenated_x.append(x)
    
        return torch.cat(concatenated_x, dim=1)
    
class TransitionLayer(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int
                 ):
        
        """
        out dimension is decided by compression factor (Theta)
        if compression factor < 1, then, the model is dubbed DenseNet-C
        compression factor set to 0.5 based on the official implementation
        See section 3.Compression in https://arxiv.org/abs/1608.06993
        """
        layers = [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ]

        super().__init__(*layers)


class DenseNet(nn.Module):
    def __init__(self,
                 block_list: list[int],
                 num_classes: int,
                 growth_rate: int,
                 compression_factor: float,
                 bottleneck: bool,
                 is_data_small: bool
                 ):
        
        super().__init__()

        self.len_block_list = len(block_list)
        
        if is_data_small:
            # BC model use 2 * growth rate
            in_channels = 2 * growth_rate if bottleneck else 16
            self.conv = nn.Sequential(
                nn.Conv2d(3, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        
        else:
            in_channels = 2 * growth_rate
            self.conv = nn.Sequential(
                nn.Conv2d(3, in_channels, kernel_size=7, padding=3, stride=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )

        
        # add dense blocks and transition layer
        for i in range(self.len_block_list):
            dense_block = DenseBlock(
                in_channels=in_channels,
                num_blocks=block_list[i],
                growth_rate=growth_rate,
                bottleneck=bottleneck
            )
            setattr(self, f"dense{i+1}", dense_block)

            # update channel
            in_channels += block_list[i] * growth_rate

            if i != self.len_block_list - 1: # classifier is applied to output of dense block
                out_channels = int(in_channels * compression_factor)

                transition_layer = TransitionLayer(in_channels, out_channels)
                setattr(self, f"transition{i+1}", transition_layer)

                # output channels is compressed by factor
                in_channels = out_channels

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            """
            He Normal Initialization
            - initialization is based on https://arxiv.org/abs/1502.01852
            - normal gaussian with sqrt(2. / out units) std with zero mean
            """
            #
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        for i in range(self.len_block_list):
            x = getattr(self, f'dense{i+1}')(x)

            if i != self.len_block_list - 1:
                x = getattr(self, f'transition{i+1}')(x)
        
        out = self.classifier(x)
        return out


"""
!! Different formula compared to ResNet !!

Models for small data is based on Table2 in https://arxiv.org/abs/1608.06993
- It has 3 Dense blocks with n DenseLayer, 1 conv in each 2 transition layers,
  initial 1 conv layers, and a classifier.
- [depth] = 3 * n + (2 + 1 + 1) = 3n + 4



- For DenseNet-BC, it has 1x1 conv and 3x3 conv in each n DenseLayer (2n)
- [depth] = 3 * 2 * n + (2 + 1 + 1) = 6n + 4
"""

def get_densenet(model_name: str,
                 num_classes: int,
                 growth_rate: int = None
                 ):

    model_name = model_name.lower()

    model_config_dict = {
        'densenet121': ([6, 12, 24, 16], num_classes, 32, 0.5, True, False),
        'densenet168': ([6, 12, 32, 32], num_classes, 32, 0.5, True, False),
        'densenet201': ([6, 12, 48, 32], num_classes, 32, 0.5, True, False),
        'densenet264': ([6, 12, 64, 48], num_classes, 32, 0.5, True, False), # To this line, model for ImageNet
        'densenet40':  ([12, 12, 12], num_classes, 12, 1.0, False, True), # from this line, model for small data (Table 2)
        'densenet100':  ([32, 32, 32], num_classes, growth_rate, 1.0, False, True),
        'densenet100bc':  ([16, 16, 16], num_classes, 12, 0.5, True, True),
        'densenet250bc':  ([41, 41, 41], num_classes, 24, 0.5, True, True),
        'densenet190bc':  ([31, 31, 31], num_classes, 40, 0.5, True, True)
    }

    # special case for DenseNet100 k=12 or k=24
    if model_name == 'densenet100' and growth_rate not in [12, 24]:
        raise ValueError(f"Growth rate must be one of [12, 24]. Got {growth_rate}")
    if model_name not in model_config_dict.keys():
        raise ValueError(f"Given model name does not exit in resnet model config. Got {model_name}")

    model_config = model_config_dict[model_name]
    return DenseNet(block_list=model_config[0],
                    num_classes=model_config[1],
                    growth_rate=model_config[2],
                    compression_factor=model_config[3],
                    bottleneck=model_config[4],
                    is_data_small=model_config[5]
                    )


if __name__ == '__main__':
    import torch

    def model_summary(model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p  in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


    device = 'cpu'
    model = get_densenet('densenet250bc', 10, 12).to(device) # for comparison with # of params in table 6

    model_summary(model)

    """
    with torch.no_grad():
        model.eval()

        data = torch.randn(2, 3, 224, 224).to(device)
        output = model(data)
        pred = output.argmax(dim=-1)

        print(f"Output shape: {output.shape}")
        print(f"Predictions: {pred}")
    """
    ###################################
    # Complete Model Checking