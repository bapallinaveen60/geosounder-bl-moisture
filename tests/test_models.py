"""Unit tests for ConvLSTM and ThermoBL-Net neural architectures."""

import torch

from src.models.baselines.point_mlp import PointRadianceMLP
from src.models.baselines.unet_2d import StaticUNet2D
from src.models.convlstm import ConvLSTM, ConvLSTMCell
from src.models.thermobonet import ThermoBLNet


def test_convlstm_cell_forward():
    cell = ConvLSTMCell(in_channels=16, hidden_dim=32, kernel_size=3, peephole=True)
    x = torch.randn(2, 16, 32, 32)
    h, c = cell(x)

    assert h.shape == (2, 32, 32, 32)
    assert c.shape == (2, 32, 32, 32)

    # Test recurrent step with existing state
    h2, c2 = cell(x, (h, c))
    assert h2.shape == (2, 32, 32, 32)
    assert c2.shape == (2, 32, 32, 32)


def test_convlstm_multilayer_forward():
    model = ConvLSTM(
        in_channels=16,
        hidden_dims=[32, 16],
        kernel_size=3,
        num_layers=2,
        peephole=True,
    )
    # Batch size 2, 4 time steps, 16 channels, 32x32 spatial grid
    x_seq = torch.randn(2, 4, 16, 32, 32)
    out_seq, last_states = model(x_seq)

    assert out_seq.shape == (2, 4, 16, 32, 32)
    assert len(last_states) == 2


def test_thermobonet_end_to_end():
    model = ThermoBLNet(
        in_channels=26,
        out_levels=5,
        conv3d_features=32,
        convlstm_hidden=(32, 32),
    )
    # Batch size 2, 4 hours, 26 channels, 64x64 grid
    x = torch.randn(2, 4, 26, 64, 64)
    out = model(x)

    # Must produce 5 levels: [1000, 950, 925, 850, 700] hPa
    assert out.shape == (2, 5, 64, 64)
    # Specific humidity must be strictly non-negative
    assert torch.all(out >= 0.0)

    # Verify backpropagation gradients
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"


def test_baselines_forward():
    # Point MLP
    mlp = PointRadianceMLP(in_features=26, out_levels=5)
    x_spatial = torch.randn(2, 26, 16, 16)
    out_mlp = mlp(x_spatial)
    assert out_mlp.shape == (2, 5, 16, 16)

    # Static 2D U-Net
    unet = StaticUNet2D(in_channels=26, out_levels=5)
    out_unet = unet(x_spatial)
    assert out_unet.shape == (2, 5, 16, 16)
