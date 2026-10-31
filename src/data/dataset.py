"""PyTorch Dataset & DataLoader for ThermoBL-Net.

Assembles 4-hour temporal rolling sequences (t-3, t-2, t-1, t) across 26 multi-modal channels
and generates target 3D specific humidity volumetric fields across 5 vertical pressure levels
[1000, 950, 925, 850, 700] hPa alongside GNSS PWV and radiosonde collocation masks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ThermoBLDataset(Dataset):
    """Spatiotemporal rolling window dataset for boundary-layer thermodynamic retrieval."""

    def __init__(
        self,
        num_samples: int = 100,
        temporal_window: int = 4,
        in_channels: int = 26,
        num_levels: int = 5,
        grid_shape: Tuple[int, int] = (128, 128),
        is_synthetic: bool = True,
    ) -> None:
        """Initialize Dataset.

        Args:
            num_samples: Number of temporal sequence windows.
            temporal_window: Number of consecutive time steps (K=4 hours).
            in_channels: Number of input variables per time step (26 channels).
            num_levels: Number of output pressure levels (5 levels).
            grid_shape: Spatial tile dimensions (H, W).
            is_synthetic: Flag for generating realistic synthetic data when raw store is unavailable.
        """
        self.num_samples = num_samples
        self.temporal_window = temporal_window
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.grid_shape = grid_shape
        self.is_synthetic = is_synthetic

        # Standard pressure levels: [1000, 950, 925, 850, 700] hPa
        self.pressure_levels = np.array([1000.0, 950.0, 925.0, 850.0, 700.0], dtype=np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch a single sample sequence.

        Returns:
            Dictionary containing:
                - 'input_sequence': Tensor of shape (K=4, C=26, H=128, W=128)
                - 'target_q': Tensor of shape (L=5, H=128, W=128) in g/kg
                - 'target_mask': Boolean mask for collocated radiosondes (L=5, H=128, W=128)
                - 'gnss_pwv': Ground truth GNSS PWV map (H=128, W=128) in mm
                - 'pwv_mask': Boolean mask for active GNSS receiver stations (H=128, W=128)
                - 'winds_uv': Horizontal surface wind vectors (2, H=128, W=128) in m/s
                - 'upper_pwv': Upper troposphere PWV residual above 700 hPa (H=128, W=128)
        """
        h, w = self.grid_shape
        k = self.temporal_window
        c = self.in_channels
        l = self.num_levels

        if self.is_synthetic:
            # Generate physically realistic synthetic tensors for Eastern India pre-monsoon:
            # Sounder brightness temps: 240 K to 310 K
            # GNSS PWV: 35 mm to 65 mm (maritime moist advection vs dry continental air)
            # Surface specific humidity: 14 to 22 g/kg at 1000 hPa, tapering to 3 to 6 g/kg at 700 hPa
            input_seq = np.random.uniform(250.0, 310.0, size=(k, c, h, w)).astype(np.float32)

            # Synthetic moisture profile: decrease monotonically with height
            base_q = np.linspace(18.0, 4.0, num=l).astype(np.float32)  # shape (5,)
            spatial_grad = np.linspace(-3.0, 3.0, num=w, dtype=np.float32).reshape(1, 1, w)
            target_q = np.broadcast_to(base_q.reshape(l, 1, 1), (l, h, w)).copy() + spatial_grad
            target_q = np.maximum(0.5, target_q)

            # Random radiosonde collocation point
            target_mask = np.zeros((l, h, w), dtype=bool)
            target_mask[:, h // 2, w // 2] = True

            # Invariant GNSS PWV
            gnss_pwv = np.random.uniform(40.0, 58.0, size=(h, w)).astype(np.float32)
            pwv_mask = np.random.binomial(1, 0.05, size=(h, w)).astype(bool)

            # Horizontal winds (u: westerlies/easterlies, v: southerlies maritime surge)
            winds_u = np.full((1, h, w), 5.0, dtype=np.float32)
            winds_v = np.full((1, h, w), 8.0, dtype=np.float32)
            winds_uv = np.concatenate([winds_u, winds_v], axis=0)

            # Upper troposphere PWV residual above 700 hPa (~10-15 mm)
            upper_pwv = np.full((h, w), 12.0, dtype=np.float32)

            return {
                "input_sequence": torch.from_numpy(input_seq),
                "target_q": torch.from_numpy(target_q),
                "target_mask": torch.from_numpy(target_mask),
                "gnss_pwv": torch.from_numpy(gnss_pwv),
                "pwv_mask": torch.from_numpy(pwv_mask),
                "winds_uv": torch.from_numpy(winds_uv),
                "upper_pwv": torch.from_numpy(upper_pwv),
            }

        raise NotImplementedError("Live HDF5/Zarr reader attached during pipeline assembly.")


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """Create PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
