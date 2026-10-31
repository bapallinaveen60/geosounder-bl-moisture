"""Static 2D U-Net Benchmark Baseline.

Processes single static hourly observation frames without temporal recurrent memory
to quantify the explicit value of multi-hour ConvLSTM temporal dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> GroupNorm -> LeakyReLU) * 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class StaticUNet2D(nn.Module):
    """Standard 2D U-Net applied to static hourly frames."""

    def __init__(self, in_channels: int = 26, out_levels: int = 5) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_levels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Single time-frame input tensor (B, C=26, H, W) or (B, Time, C, H, W).
           If 5D sequence is passed, the final time step is extracted.
        """
        if x.ndim == 5:
            x = x[:, -1, :, :, :]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        x_u1 = self.conv_up1(u1)

        u2 = self.up2(x_u1)
        u2 = torch.cat([u2, x1], dim=1)
        x_u2 = self.conv_up2(u2)

        logits = self.outc(x_u2)
        return F.relu(logits)
