"""ThermoBL-Net: Thermodynamic Boundary-Layer Sounder Neural Network.

A Physics-Constrained Spatiotemporal Deep Learning Model fusing hourly INSAT Sounder radiances,
ground GNSS PWV, surface meteorology, and topography to reconstruct 3D boundary-layer specific
humidity volumetric fields across [1000, 950, 925, 850, 700] hPa.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm import ConvLSTM


class Spatiotemporal3DEncoder(nn.Module):
    """3D-Convolutional feature extractor for multi-temporal satellite sequences."""

    def __init__(self, in_channels: int = 26, out_channels: int = 64) -> None:
        super().__init__()
        # Ingests (B, C_in, Time, H, W)
        self.conv3d_1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            padding_mode="replicate",
        )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        self.conv3d_2 = nn.Conv3d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            padding_mode="replicate",
        )
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Args:

        x_seq: (B, Time, C_in, H, W).

        Returns:
            Features: (B, Time, Out_channels, H, W).
        """
        # Permute to (B, C_in, Time, H, W) for 3D convolution
        x = x_seq.permute(0, 2, 1, 3, 4)
        x = F.leaky_relu(self.norm1(self.conv3d_1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.norm2(self.conv3d_2(x)), negative_slope=0.1)
        # Permute back to (B, Time, Out_channels, H, W)
        return x.permute(0, 2, 1, 3, 4)


class MultiScaleVolumetricDecoder(nn.Module):
    """Decodes latent hidden state into 3D volumetric moisture fields at 5 pressure levels."""

    def __init__(
        self,
        in_channels: int = 64,
        out_levels: int = 5,
        hidden_channels: Tuple[int, int] = (64, 32),
    ) -> None:
        super().__init__()
        self.out_levels = out_levels

        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=hidden_channels[0])

        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=hidden_channels[1])

        # Final projection to L vertical pressure levels
        self.final_proj = nn.Conv2d(hidden_channels[1], out_levels, kernel_size=1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Args:

        hidden_state: Latent representation at final time step (B, C, H, W).

        Returns:
            q_hat: Reconstructed specific humidity in g/kg (B, L=5, H, W).
        """
        x = F.leaky_relu(self.norm1(self.conv1(hidden_state)), negative_slope=0.1)
        x = F.leaky_relu(self.norm2(self.conv2(x)), negative_slope=0.1)
        out = self.final_proj(x)
        # ReLU activation ensures non-negative atmospheric specific humidity (q >= 0)
        return F.relu(out)


class ThermoBLNet(nn.Module):
    """End-to-End Thermodynamic Boundary-Layer Sounder Network."""

    def __init__(
        self,
        in_channels: int = 26,
        out_levels: int = 5,
        conv3d_features: int = 64,
        convlstm_hidden: Tuple[int, ...] = (128, 128, 64),
        kernel_size: int = 3,
        dropout_prob: float = 0.1,
    ) -> None:
        """Initialize ThermoBL-Net.

        Args:
            in_channels: Total input channel count (default: 26).
            out_levels: Number of pressure levels (default: 5 [1000, 950, 925, 850, 700 hPa]).
            conv3d_features: Channel capacity of 3D-Conv spatiotemporal encoder.
            convlstm_hidden: Tuple of hidden dimensions across ConvLSTM layers.
            kernel_size: Spatial kernel size.
            dropout_prob: Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_levels = out_levels

        # 1. Spatiotemporal feature extraction
        self.encoder = Spatiotemporal3DEncoder(
            in_channels=in_channels,
            out_channels=conv3d_features,
        )

        # 2. Recurrent ConvLSTM backbone
        self.convlstm = ConvLSTM(
            in_channels=conv3d_features,
            hidden_dims=list(convlstm_hidden),
            kernel_size=kernel_size,
            num_layers=len(convlstm_hidden),
            peephole=True,
            dropout_prob=dropout_prob,
        )

        # 3. Multi-scale volumetric decoder
        self.decoder = MultiScaleVolumetricDecoder(
            in_channels=convlstm_hidden[-1],
            out_levels=out_levels,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_seq: (B, Time=4, C_in=26, H, W).

        Returns:
            q_pred: Reconstructed specific humidity profile tensor (B, L=5, H, W) in g/kg.
        """
        # Spatiotemporal 3D encoding
        feat_seq = self.encoder(x_seq)  # (B, Time, 64, H, W)

        # Recurrent temporal evolution
        out_seq, last_states = self.convlstm(feat_seq)

        # Latent state at the current observation time step t (final step of sequence)
        h_final = out_seq[:, -1, :, :, :]  # (B, hidden_dim, H, W)

        # Volumetric decoding to 5 standard boundary-layer levels
        q_pred = self.decoder(h_final)  # (B, 5, H, W)

        return q_pred
