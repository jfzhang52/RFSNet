"""
Raw version of Recurrent Fine-grained Self-Attention Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange

from src.misc.utilities import Config
from src.misc.nn import Conv2D, SelfAttention


class RFSNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super(RFSNet, self).__init__()
        num_blocks = cfg.model.num_blocks if cfg.model.num_blocks else 4
        use_bn = cfg.model.use_bn if cfg.model.use_bn else False
        lw = cfg.model.lw if cfg.model.lw else True

        # Encoder
        self.Backbone = models.vgg16(pretrained=True).features[:23]

        # Decoder
        self.Blocks = nn.ModuleList([SABlock(use_bn=use_bn, lw=lw)] * num_blocks)

        # Regression head
        self.Head = nn.Sequential(
            Conv2D(512, 128, 3, 1, activation=nn.ReLU(), use_bn=use_bn),
            Conv2D(128, 64, 3, 1, activation=nn.ReLU(), use_bn=use_bn),
            Conv2D(64, 1, 1))

    def forward(self, x: torch.Tensor) -> dict:
        # Resize & Adjust channels
        B, T, C, H, W = x.shape                 # [B, T, C, H, W]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = F.interpolate(x, (360, 640), mode='bilinear', align_corners=False)

        # Backbone
        x = self.Backbone(x)                    # [B * T, 512, H / 8, W / 8]
        encoder_output = x

        # Dense Blocks
        x = rearrange(x, '(b t) c h w -> b c t h w', t=T)
        for block in self.Blocks:
            x = block(x)                        # [B, 512, T, h, w]
        decoder_output = x

        # Regression Head
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.Head(x)                        # [B * T, 1, h, w]

        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=T)

        return {'density': x,
                'encoder_output': encoder_output,
                'decoder_output': decoder_output}


class SABlock(nn.Module):
    def __init__(self,
                 dim: int = 512,
                 use_bn: bool = False,
                 patch_size: int = 5,
                 reduction: int = 4,
                 lw: bool = True) -> None:

        super(SABlock, self).__init__()

        self.SpatialSA = RegionalSpatialBooster(
            dim_in=dim, patch_size=patch_size, reduction=reduction)
        self.gamma1 = nn.Parameter(torch.zeros(1), requires_grad=lw)

        self.TemporalSA = GlobalTemporalBooster(
            dim_in=dim, reduction=reduction)
        self.gamma2 = nn.Parameter(torch.zeros(1), requires_grad=lw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape                 # [B, 512, T, H, W]

        # spatial self-attention
        xs = rearrange(x, 'b c t h w -> (b t) h w c')
        xs = self.SpatialSA(xs)                 # [B * T, H, W, 512]
        xs = rearrange(xs, '(b t) h w c -> b c t h w', t=T)
        x = x + self.gamma1 * xs                # [B, 512, T, H, W]

        # temporal self-attention
        xt = rearrange(x, 'b c t h w -> (b h w) t c')
        xt = self.TemporalSA(xt)                # [B * H * W, T, 512]
        xt = rearrange(xt, '(b h w) t c -> b c t h w', h=H, w=W)
        x = x + self.gamma2 * xt                # [B, 512, T, H, W]

        return x


class RegionalSpatialBooster(nn.Module):
    def __init__(self,
                 dim_in: int,
                 num_heads: int = 8,
                 dim_head: int = 16,
                 dropout: float = 0.,
                 patch_size: int = 5,
                 reduction: int = 4) -> None:

        super(RegionalSpatialBooster, self).__init__()

        self.ph, self.pw = patch_size, patch_size

        self.attn = SelfAttention(dim_in, dim_head, num_heads, dropout)

        embed_dim = dim_in // reduction
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, dim_in))

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, H, W, C = x.shape            # [N, H, W, C]

        # 1. Local SA
        x = rearrange(x, 'n (nh ph) (nw pw) c -> (n nh nw) (ph pw) c', ph=self.ph, pw=self.pw)
        x = self.drop(self.attn(x))
        x = rearrange(x, '(n nh nw) (ph pw) c -> n (nh ph) (nw pw) c',
                      ph=self.ph, pw=self.pw, nh=H // self.ph, nw=W // self.pw)

        # 2. MLP
        x = self.drop(self.mlp(x))

        return x


class GlobalTemporalBooster(nn.Module):
    def __init__(self,
                 dim_in: int,
                 num_heads: int = 8,
                 dim_head: int = 16,
                 dropout: float = 0.,
                 reduction: int = 4) -> None:

        super(GlobalTemporalBooster, self).__init__()

        self.attn = SelfAttention(dim_in, dim_head, num_heads, dropout)

        embed_dim = dim_in // reduction
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, dim_in))

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, C = x.shape           # [N, T, C]

        # 1. Self-Attention
        x = self.drop(self.attn(x))

        # 2. MLP
        x = self.drop(self.mlp(x))

        return x
