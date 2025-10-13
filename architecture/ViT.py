import torch.nn as nn
import torch
import math
from collections import OrderedDict

class FFN(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 drop_p: float
                 ):

        hidden_channels = in_channels * 4
        layers = [
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(drop_p) if drop_p > 0 else nn.Identity(),
            nn.Linear(hidden_channels, in_channels)
        ]

        super().__init__(*layers) 


class MHSA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int
                 ):
        super().__init__()

        self.num_heads = num_heads
        self.head_channels = in_channels // num_heads
        self.scale = self.head_channels**0.5

        self.ln = nn.LayerNorm(in_channels)

        self.proj_q = nn.Linear(in_channels, in_channels)
        self.proj_k = nn.Linear(in_channels, in_channels)
        self.proj_v = nn.Linear(in_channels, in_channels)

        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)
        B, L, C = x.shape

        x = self.ln(x)

        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.reshape(B, L, self.num_heads, self.head_channels).permute(0, 2, 1, 3) # (B, H, L, D)
        k = k.reshape(B, L, self.num_heads, self.head_channels).permute(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_channels).permute(0, 2, 1, 3)

        attn_weight = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)
        attn = torch.matmul(attn_weight, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, L, C)

        out = self.fc(attn)
        return out
    
class ViTBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 drop_p: float
                 ):
        
        super().__init__()

        self.drop_p = drop_p

        self.mhsa = MHSA(in_channels, num_heads)
        self.ffn = FFN(in_channels, drop_p)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)

        residual = x
        out = self.mhsa(x)
        out = self.dropout(out)
        out += residual

        residual = out
        out = self.ffn(out)
        out = self.dropout(out)
        out += residual
        return out
    

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 seq_len: int,
                 num_blocks: int,
                 num_heads: int,
                 drop_p: float
                 ):
        
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, in_channels))
        self.layers = nn.ModuleList([ViTBlock(in_channels, num_heads, drop_p) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor):
        x += self.pos_embed
        out = self.dropout(x)

        for layer in self.layers:
            out = layer(out)

        return out
    

class ViT(nn.Module):
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 num_blocks: int,
                 num_heads:int,
                 num_classes:int,
                 drop_p: float = 0.0,
                 img_size: int = 224,
                 fine_tuning: bool = False
                 ):
        
        super().__init__()

        seq_len = (img_size // patch_size)**2 + 1 # 1 for [cls]

        self.patch_embed = nn.Conv2d(3, in_channels, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.encoder = Encoder(in_channels, seq_len, num_blocks, num_heads, drop_p)
        self.ln = nn.LayerNorm(in_channels)

        head_layers = OrderedDict()
        if not fine_tuning: 
            head_layers["pre_logits"] = nn.Linear(in_channels, in_channels)
            head_layers["act"] = nn.Tanh()
            head_layers["head"] = nn.Linear(in_channels, num_classes)
            self.heads = nn.Sequential(head_layers)
        
        else:
            head_layers["head"] = nn.Linear(in_channels, num_classes)
            self.heads = nn.Sequential(head_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # As there are no any mentions about weight initialization in the ViT paper
        # I follow torchvision's ViT weight initialization method.
        # https://github.com/pytorch/vision/blob/01b9faa16cfeacbb70aa33bd18534de50891786b/torchvision/models/vision_transformer.py#L245
        if isinstance(module, nn.Conv2d) and module == self.patch_embed:
            fan_in = self.patch_embed.in_channels * self.patch_embed.kernel_size[0] * self.patch_embed.kernel_size[1]
            nn.init.trunc_normal_(self.patch_embed.weight, std=math.sqrt(1 / fan_in))
            if self.patch_embed.bias is not None:
                nn.init.zeros_(self.patch_embed.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)

        patch = self.patch_embed(x) # (B, C, H//P, W//P)
        patch = patch.flatten(2).transpose(2, 1) # (B, L, C)

        cls_token = self.cls_token.expand(patch.shape[0], -1, -1)
        out = torch.cat((cls_token, patch), dim=1)

        out = self.encoder(out)
        cls = out[:, 0, :]
        cls = self.ln(cls)
        preds = self.heads(cls)
        return preds
    


def get_vit(model_name: str,
            num_classes: int,
            img_size: int = 32,
            fine_tuning: bool = False
            ):

    model_config_dict = {
        'vitbase16':  (768, 16, 12, 12),
        'vitbase32':  (768, 32, 12, 12),
        'vitlarge16':  (1024, 16, 24, 16),
        'vitlarge32':  (1024, 32, 24, 16),
        'vithuge14':  (1280, 14, 32, 16),
        'cifarvit': (768, 4, 12, 12)
    }

    # special case for DenseNet100 k=12 or k=24
    if model_name not in model_config_dict.keys():
        raise ValueError(f"Given model name does not exit in ViT model config. Got {model_name}")

    model_config = model_config_dict[model_name]
    return ViT(
        in_channels=model_config[0],
        patch_size=model_config[1],
        num_blocks=model_config[2],
        num_heads=model_config[3],
        num_classes=num_classes,
        drop_p=0.0,
        img_size=img_size,
        fine_tuning=fine_tuning
    )


if __name__ == '__main__':
    import torch

    def model_summary(model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p  in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


    device = 'cpu'
    model = get_vit('vitbase32', 1000, 224).to(device) # for comparison with # of params in table 6

    model_summary(model)

    
    with torch.no_grad():
        model.eval()

        data = torch.randn(2, 3, 224, 224).to(device)
        output = model(data)
        pred = output.argmax(dim=-1)

        print(f"Output shape: {output.shape}")
        print(f"Predictions: {pred}")

    ###################################
    # Complete Model Checking