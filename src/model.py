import torch
import torch.nn as nn
from typing import List

class ConvBlock(nn.Module):
    """(Conv ► BN ► GELU) × 2  + optional Dropout."""
    def __init__(self, in_c: int, out_c: int, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Dropout2d(p_drop) if p_drop else nn.Identity(),
        )

    def forward(self, x): return self.net(x)


class AttentionGate(nn.Module):
    """
    1×1 convolutions create query (g) and key (x) vectors,
    then attention weights refine the skip connection.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        psi = self.psi(nn.GELU()(g1 + x1))
        return x * psi  # element‑wise modulation


class Down(nn.Module):
    """Down‑sampling step: ConvBlock → 2×2 MaxPool."""
    def __init__(self, in_c, out_c, p_drop):
        super().__init__()
        self.block = ConvBlock(in_c, out_c, p_drop)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x): 
        x = self.block(x)
        return x, self.pool(x)


class Up(nn.Module):
    """Up‑sample → AttentionGate → concat → ConvBlock."""
    def __init__(self, in_c, out_c, p_drop):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.att = AttentionGate(F_g=out_c, F_l=out_c, F_int=out_c // 2)
        self.conv = ConvBlock(in_c, out_c, p_drop)

    def forward(self, g, x):
        g = self.up(g)
        x = self.att(g, x)
        return self.conv(torch.cat([g, x], dim=1))


class AttentionUNet(nn.Module):
    """
    Inputs:
        channels_in  – number of meteorological variables per grid cell (e.g., 3‑D fields ➜ C, H, W)
        channels_out – number of target variables (e.g., downscaled T, RH, wind)
    """
    def __init__(
        self,
        channels_in: int = 5,
        channels_out: int = 5,
        width_factors: List[int] = [64, 128, 256, 512],
        p_drop: float = 0.1,
    ):
        super().__init__()
        wf = width_factors
        self.down1 = Down(channels_in,  wf[0], p_drop)
        self.down2 = Down( wf[0],       wf[1], p_drop)
        self.down3 = Down( wf[1],       wf[2], p_drop)

        self.bottom = ConvBlock(wf[2],  wf[3], p_drop)

        self.up3   = Up(wf[3], wf[2], p_drop)
        self.up2   = Up(wf[2], wf[1], p_drop)
        self.up1   = Up(wf[1], wf[0], p_drop)

        self.head  = nn.Conv2d(wf[0], channels_out, kernel_size=1)

    def forward(self, x):
        x1, p1 = self.down1(x)
        x2, p2 = self.down2(p1)
        x3, p3 = self.down3(p2)

        btm = self.bottom(p3)

        u3 = self.up3(btm, x3)
        u2 = self.up2(u3,  x2)
        u1 = self.up1(u2,  x1)

        return self.head(u1)
