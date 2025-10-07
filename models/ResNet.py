import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: nn.Module = None
                 ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU()

        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.layers(x)

        if self.downsample is not None: # in case of need projection in skip connection
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x
    

class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: nn.Module = None
                 ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 4 * out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(4 * out_channels)
        )

        self.relu = nn.ReLU()

        self.downsample = downsample


    def forward(self, x: torch.Tensor):
        residual = x
        x = self.layers(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x
    

class ResNetBlock(nn.Sequential):
    def __init__(self,
                 block:nn.Module,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 stride: int,
                 expansion: int
                 ):
        

        downsample = None
        if stride != 1 or in_channels != out_channels * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * expansion)
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))


        # after first block, in_channels is changed to out_channels * expansion
        for _ in range(num_blocks-1):
            layers.append(block(out_channels * expansion, out_channels, stride=1, downsample=None))

        # as it is inherited from nn.Sequential, forward() method automatically defined
        super().__init__(*layers)
    

class ResNet(nn.Module):
    def __init__(self,
                 block_list: list[int],
                 num_classes:int,
                 bottleneck: bool,
                 is_data_small:bool):
        super().__init__()


        if bottleneck:
            expansion = 4
            block = Bottleneck
        else:
            expansion = 1
            block = ResidualBlock


        if is_data_small:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )

            self.conv2 = ResNetBlock(block, 16, 16,
                                     block_list[0], stride=1, expansion=expansion) 
            self.conv3 = ResNetBlock(block, 16 * expansion, 32,
                                     block_list[1], stride=2, expansion=expansion)
            self.conv4 = ResNetBlock(block, 32 * expansion, 64,
                                     block_list[2], stride=2, expansion=expansion)
            self.conv5 = None
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64 * expansion, num_classes)
            )

        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )

            self.conv2 = ResNetBlock(block, 64, 64,
                                     block_list[0], stride=1, expansion=expansion) 
            self.conv3 = ResNetBlock(block, 64 * expansion, 128,
                                     block_list[1], stride=2, expansion=expansion)
            self.conv4 = ResNetBlock(block, 128 * expansion, 256,
                                     block_list[2], stride=2, expansion=expansion)
            self.conv5 = ResNetBlock(block, 256 * expansion, 512,
                                     block_list[3], stride=2, expansion=expansion)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512 * expansion, num_classes)
            )

        # nn.Module.apply() recursively applies the given function to every submodule
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
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


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.conv5 is not None:
            x = self.conv5(x)

        out = self.classifier(x)
        return out
    

def get_resnet(model_name: str, num_classes: int, is_data_small: bool):
    
    model_config_dict = {
        'resnet18': ([2, 2, 2, 2], False,is_data_small),
        'resnet34': ([3, 4, 6, 3], False, is_data_small),
        'resnet50': ([3, 4, 6, 3], True, is_data_small),
        'resnet101': ([3, 4, 23, 3], True, is_data_small),
        'resnet152': ([3, 8, 36, 3], True, is_data_small),
        'resnet20': ([3, 3, 3], False, True), # for small data from this line
        'resnet32': ([5, 5, 5], False, True),
        'resnet44': ([7, 7, 7], False, True),
        'resnet56': ([9, 9, 9], False, True),
        'resnet68': ([11, 11, 11], False, True),
        'resnet110': ([18, 18, 18], False, True),
        'resnet1202': ([200, 200, 200], False, True)
    }

    if model_name not in model_config_dict.keys():
        raise ValueError(f"Given model name does not exit in ResNet model config. Got {model_name}")

    model_config = model_config_dict[model_name]
    return ResNet(model_config[0], num_classes, model_config[1], model_config[2])


if __name__ == '__main__':
    import torch

    def model_summary(model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p  in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


    model = get_resnet('resnet44', 10, True) # for comparison with # of params in table 6

    model_summary(model)

    with torch.no_grad():
        device = 'cpu'
        model.eval()

        data = torch.randn(2, 3, 32, 32).to(device)
        output = model(data)
        pred = output.argmax(dim=-1)

        print(f"Output shape: {output.shape}")
        print(f"Predictions: {pred}")

    ###################################
    # Complete Model Checking






        


    
    

        


    