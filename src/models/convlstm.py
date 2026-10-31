"""Convolutional Long Short-Term Memory (ConvLSTM) Core Recurrent Neural Network.

Implements spatial convolution operators instead of standard linear multiplications
within the recurrent cell to preserve spatial correlation structures of boundary-layer
moisture pooling, drylines, and orographic convergence zones.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single-layer Convolutional LSTM cell with peephole connections and GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
        peephole: bool = True,
        padding_mode: str = "replicate",
        num_groups: int = 8,
    ) -> None:
        """Initialize ConvLSTM Cell.

        Args:
            in_channels: Number of input channels.
            hidden_dim: Number of hidden state feature maps.
            kernel_size: Size of convolutional kernel (default: 3).
            bias: Whether to add learnable bias.
            peephole: Enable peephole connections from cell state C to gates.
            padding_mode: Spatial padding mode ('zeros', 'reflect', 'replicate', 'circular').
            num_groups: Group count for GroupNorm.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.peephole = peephole

        # Combined convolution for input and candidate gates: [i, f, c, o]
        self.conv_x = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.conv_h = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            padding_mode=padding_mode,
            bias=bias,
        )

        # Normalization layer for recurrent hidden transitions
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=4 * hidden_dim)

        # Peephole weight parameters
        if self.peephole:
            self.w_ci = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
            self.w_cf = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
            self.w_co = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        else:
            self.register_parameter("w_ci", None)
            self.register_parameter("w_cf", None)
            self.register_parameter("w_co", None)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one recurrent step.

        Args:
            x: Input tensor of shape (B, C_in, H, W).
            state: Tuple of (H_{t-1}, C_{t-1}), each of shape (B, hidden_dim, H, W).

        Returns:
            Tuple of (H_t, C_t).
        """
        batch_size, _, height, width = x.shape

        if state is None:
            h_prev = torch.zeros(
                batch_size, self.hidden_dim, height, width, device=x.device, dtype=x.dtype
            )
            c_prev = torch.zeros(
                batch_size, self.hidden_dim, height, width, device=x.device, dtype=x.dtype
            )
        else:
            h_prev, c_prev = state

        # Compute affine convolutions
        conv_gates = self.conv_x(x) + self.conv_h(h_prev)
        conv_gates = self.norm(conv_gates)

        # Split into four gate components
        cc_i, cc_f, cc_c, cc_o = torch.chunk(conv_gates, 4, dim=1)

        if self.peephole:
            i_t = torch.sigmoid(cc_i + self.w_ci * c_prev)
            f_t = torch.sigmoid(cc_f + self.w_cf * c_prev)
        else:
            i_t = torch.sigmoid(cc_i)
            f_t = torch.sigmoid(cc_f)

        c_tilde = torch.tanh(cc_c)
        c_next = f_t * c_prev + i_t * c_tilde

        if self.peephole:
            o_t = torch.sigmoid(cc_o + self.w_co * c_next)
        else:
            o_t = torch.sigmoid(cc_o)

        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """Multi-layer Spatiotemporal ConvLSTM sequence processor."""

    def __init__(
        self,
        in_channels: int,
        hidden_dims: List[int],
        kernel_size: int = 3,
        num_layers: int = 3,
        peephole: bool = True,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            cur_in = in_channels if i == 0 else hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    in_channels=cur_in,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_size,
                    peephole=peephole,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(
        self,
        x_seq: torch.Tensor,
        init_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Process a sequence of spatial tensors over time.

        Args:
            x_seq: Tensor of shape (B, Time, C, H, W).
            init_states: Optional initial hidden and cell states per layer.

        Returns:
            Tuple of:
                - output_sequence: Hidden state of last layer across all time steps (B, Time, H_dim, H, W)
                - last_states: List of (H_t, C_t) for each layer at final time step
        """
        batch_size, time_steps, _, height, width = x_seq.shape

        current_layer_input = x_seq
        last_states = []

        for layer_idx, cell in enumerate(self.cell_list):
            layer_output_list = []
            layer_state = init_states[layer_idx] if init_states is not None else None

            for t in range(time_steps):
                x_t = current_layer_input[:, t, :, :, :]
                h_t, c_t = cell(x_t, layer_state)
                h_t = self.dropout(h_t)
                layer_state = (h_t, c_t)
                layer_output_list.append(h_t)

            last_states.append(layer_state)
            current_layer_input = torch.stack(layer_output_list, dim=1)

        return current_layer_input, last_states
