"""Unit tests for physical constraints and loss functions."""

import pytest
import torch

from src.losses.composite_loss import CompositePhysicsLoss
from src.physics.mass_conservation import ColumnMassConservationLoss, integrate_boundary_layer_pwv
from src.physics.moisture_flux import MoistureFluxConvergenceOperator
from src.physics.monotonicity import VerticalMonotonicityLoss


def test_integrate_boundary_layer_pwv():
    # Constant profile of 10 g/kg (0.010 kg/kg) across [1000, 950, 925, 850, 700] hPa
    # Delta P = (1000 - 700) * 100 Pa = 30000 Pa
    # Integral = (0.010 * 30000) / (1000 * 9.80665) = 300 / 9806.65 = 0.03059 m = 30.59 mm
    levels = [1000.0, 950.0, 925.0, 850.0, 700.0]
    q_const = torch.full((1, 5, 10, 10), 10.0, dtype=torch.float32)

    pwv = integrate_boundary_layer_pwv(q_const, pressure_levels_hpa=levels)
    expected_pwv_mm = (0.010 * 30000.0) / (1000.0 * 9.80665) * 1000.0

    assert pwv.shape == (1, 10, 10)
    assert torch.allclose(pwv, torch.tensor(expected_pwv_mm), atol=1e-2)


def test_column_mass_loss_autograd():
    criterion = ColumnMassConservationLoss()
    q_pred = torch.full((2, 5, 16, 16), 12.0, requires_grad=True)
    gnss_truth = torch.full((2, 16, 16), 45.0)
    upper_residual = torch.full((2, 16, 16), 10.0)

    loss = criterion(q_pred, gnss_truth, upper_residual)
    assert loss.item() > 0.0

    loss.backward()
    assert q_pred.grad is not None
    assert torch.all(torch.isfinite(q_pred.grad))


def test_vertical_monotonicity_loss():
    mono_loss_fn = VerticalMonotonicityLoss(delta_q=0.2)

    # 1. Perfectly monotonically decreasing profile (Surface=18 g/kg down to 700 hPa=4 g/kg)
    q_monotonic = torch.tensor([18.0, 14.0, 11.0, 7.0, 4.0]).view(1, 5, 1, 1)
    loss_zero = mono_loss_fn(q_monotonic)
    assert loss_zero.item() == 0.0

    # 2. Inverted profile (Surface=10 g/kg, sharp inversion to 16 g/kg at 950 hPa)
    q_inverted = torch.tensor([10.0, 16.0, 11.0, 7.0, 4.0]).view(1, 5, 1, 1)
    loss_penalized = mono_loss_fn(q_inverted)
    # diff = 16.0 - 10.0 - 0.2 = 5.8
    assert torch.isclose(loss_penalized, torch.tensor(5.8 / 4.0), atol=1e-3)


def test_moisture_flux_convergence_operator():
    mfc_op = MoistureFluxConvergenceOperator(dx_meters=4000.0, dy_meters=4000.0)
    q = torch.full((1, 1, 32, 32), 15.0)
    u = torch.full((1, 1, 32, 32), 5.0)
    v = torch.full((1, 1, 32, 32), 5.0)

    # Uniform field should have zero spatial divergence
    mfc_uniform = mfc_op(q, u, v)
    assert mfc_uniform.shape == (1, 1, 32, 32)
    # Interior pixels must have zero flux divergence
    assert torch.allclose(mfc_uniform[:, :, 5:-5, 5:-5], torch.zeros(1, 1, 22, 22), atol=1e-6)


def test_composite_loss_end_to_end():
    comp_loss = CompositePhysicsLoss()
    q_pred = torch.full((2, 5, 16, 16), 14.0, requires_grad=True)
    q_true = torch.full((2, 5, 16, 16), 13.5)
    gnss_truth = torch.full((2, 16, 16), 50.0)
    upper_residual = torch.full((2, 16, 16), 12.0)
    winds_uv = torch.full((2, 2, 16, 16), 4.0)

    loss_dict = comp_loss(
        q_pred=q_pred,
        q_true=q_true,
        sonde_mask=None,
        gnss_pwv_true=gnss_truth,
        upper_pwv_residual=upper_residual,
        gnss_mask=None,
        winds_uv=winds_uv,
    )

    assert "total_loss" in loss_dict
    assert "loss_profile" in loss_dict
    assert "loss_column" in loss_dict
    assert "loss_monotonic" in loss_dict
    assert "loss_conv" in loss_dict

    loss_dict["total_loss"].backward()
    assert q_pred.grad is not None
