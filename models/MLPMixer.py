import torch.nn as nn
import torch

class Mixer(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 token_mixing_channels: int,
                 channel_mixing_channels: int,
                 seq_len: int,
                 drop_p: float = 0.0
                 ):
        super().__init__()

        self.dropout = nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()

        self.ln = nn.LayerNorm(hidden_channels)
        self.token_mixing = nn.Sequential(
            nn.Linear(seq_len, token_mixing_channels),
            nn.GELU(),
            self.dropout,
            nn.Linear(token_mixing_channels, seq_len),
            self.dropout
        )

        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, channel_mixing_channels),
            nn.GELU(),
            self.dropout,
            nn.Linear(channel_mixing_channels, hidden_channels),
            self.dropout
        )

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)

        residual = x
        out = self.ln(x)
        out = out.permute(0, 2, 1)
        out = self.token_mixing(out)
        out = out.permute(0, 2, 1)
        out += residual

        residual = out
        out = self.channel_mixing(out)
        out += residual
        return out
    
class MLPMixer(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 token_mixing_channels: int,
                 channel_mixing_channels: int,
                 num_blocks: int,
                 num_classes: int,
                 patch_size: int,
                 drop_p: float = 0.0,
                 img_size: int = 224
                 ):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, hidden_channels, patch_size, stride=patch_size)
        seq_len = (img_size // patch_size) ** 2
        self.encoder = nn.Sequential(*[Mixer(hidden_channels,
                                            token_mixing_channels,
                                            channel_mixing_channels,
                                            seq_len, drop_p) for _ in range(num_blocks)])
        
        self.ln = nn.LayerNorm(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        patch = self.patch_embed(x)
        patch = patch.flatten(2).transpose(2, 1) # (B, L, C)

        out = self.encoder(patch)
        out = self.ln(out)
        out = out.mean(dim=1) # gap
        out = self.classifier(out)
        return out

    

def get_mlpmixer(model_name: str, num_classes: int, img_size: int):
    
    model_config_dict = {
        'mlpmixersmall16': (512, 256, 2048, 8, 16, 0.0),
        'mlpmixersmall32': (512, 256, 2048, 8, 32, 0.0),
        'mlpmixerbase16': (768, 384, 3072, 12, 16, 0.0),
        'mlpmixerbase32': (768, 384, 3072, 12, 32, 0.0),
        'mlpmixerlarge16': (1024, 512, 4096, 24, 16, 0.0),
        'mlpmixerlarge32': (1024, 512, 4096, 24, 32, 0.0),
        'mlpmixerhuge14': (1280, 640, 5120, 32, 14, 0.0),
    }

    if model_name not in model_config_dict.keys():
        raise ValueError(f"Given model name does not exit in MLPMixer model config. Got {model_name}")

    model_config = model_config_dict[model_name]
    return MLPMixer(model_config[0], model_config[1], model_config[2], model_config[3], num_classes, model_config[4], model_config[5], img_size)


if __name__ == '__main__':
    import torch

    def model_summary(model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p  in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


    model = get_mlpmixer('mlpmixerhuge14', 1000, 224) # for comparison with # of params in table 6

    model_summary(model)

    with torch.no_grad():
        device = 'cpu'
        model.eval()

        data = torch.randn(2, 3, 224, 224).to(device)
        output = model(data)
        pred = output.argmax(dim=-1)

        print(f"Output shape: {output.shape}")
        print(f"Predictions: {pred}")

    ###################################
    # Complete Model Checking






        


    
    

        


    