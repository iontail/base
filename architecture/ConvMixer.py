import torch
import torch.nn as nn

class Mixer(nn.Module):
    def __init__(self, in_channels:int, kernel_size:int):
        super().__init__()
        
        self.in_channels = in_channels

        self.spatial_mixing = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding='same', groups=in_channels), # depthwise conv
            nn.GELU(),
            nn.BatchNorm2d(in_channels)
        )

        self.channel_mixing = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        residual = x
        x = self.spatial_mixing(x)
        x += residual

        x = self.channel_mixing(x)
        return x
    
class Encoder(nn.Sequential):
    def __init__(self, in_channels:int, kernel_size:int, num_blocks:int):
    
        layers =[Mixer(in_channels, kernel_size) for _ in range(num_blocks) ]
        super().__init__(*layers)

class ConvMixer(nn.Module):
    def __init__(self,
                 in_channels:int,
                 num_blocks:int=8,
                 num_classes:int=1000,
                 patch_size:int=16,
                 kernel_size:int=7
                 ):

        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.kernel_size = kernel_size

        # create patch embeddings
        # (B, C, H, W) -> (B, hidden_dim, H//p, W//p)
        self.embedding = nn.Conv2d(3, in_channels, kernel_size=patch_size, stride=patch_size) 

        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(in_channels)

        self.encoder = Encoder(in_channels, kernel_size, num_blocks)

        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        x = self.embedding(x)
        x = self.gelu(x)
        x = self.bn(x)  

        x = self.encoder(x) 

        x = x.flatten(2).mean(-1)  # (B, C)
        x = self.classifier(x)
        return x
    


def get_convmixer(model_name: str, num_classes: int, img_size: int):
    
    model_config_dict = {
        'convmixer20': (1536, 20, 8, 7),
        'convmixer32': (768, 32, 7, 7),
        'cifarconvmixer': (256, 8, 2, 5)
    }

    if model_name not in model_config_dict.keys():
        raise ValueError(f"Given model name does not exit in ConvMixer model config. Got {model_name}")

    model_config = model_config_dict[model_name]
    return ConvMixer(model_config[0], model_config[1], num_classes, model_config[2], model_config[3])


if __name__ == '__main__':
    import torch

    def model_summary(model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p  in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


    model = get_convmixer('cifarconvmixer', 10, 32) # for comparison with # of params in table 6

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