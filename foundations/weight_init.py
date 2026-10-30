import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        # 2. Xavier normal standard deviation: sqrt(2 / (fan_in + fan_out))
        std = math.sqrt(2.0 / (fan_in + fan_out))

        # 3. Create tensor of shape (fan_out, fan_in) from normal distribution N(0, std^2)
        weights = torch.randn(fan_out, fan_in) * std

        # 4. Round to 4 decimal places and convert to nested list
        return torch.round(weights, decimals=4).tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        std = math.sqrt(2/fan_in)

        weights = torch.randn(fan_out, fan_in) * std

        return torch.round(weights, decimals=4).tolist()


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Use torch.manual_seed(0) once at the start.
        # Sample all weights in layer order with torch.randn(out_dim, in_dim) * std:
        # first (hidden_dim, input_dim), then (hidden_dim, hidden_dim).
        # Keep weights unrounded; do not call the init helpers, which reset the seed.
        # Then sample x = torch.randn(1, input_dim) without resetting the seed.
        # Apply x = torch.relu(x @ W.T) at every layer, with no biases.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # Use round(x.std().item(), 2) with PyTorch's default sample std.
        torch.manual_seed(0)

        # Helper to compute std given fan_in and fan_out
        def get_std(fan_in: int, fan_out: int) -> float:
            if init_type in ("xavier", "glorot"):
                return math.sqrt(2.0 / (fan_in + fan_out))
            elif init_type in ("he", "kaiming"):
                return math.sqrt(2.0 / fan_in)
            return 1.0

        # 2. Sample all weights in layer order (hidden_dim, in_dim) * std
        weights = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim
            std = get_std(fan_in, fan_out)
            w = torch.randn(fan_out, fan_in) * std
            weights.append(w)

        # 3. Sample input x without resetting the seed
        x = torch.randn(1, input_dim)

        # 4. Forward pass: x = relu(x @ W.T) and track activation std
        stds = []
        for w in weights:
            x = torch.relu(x @ w.T)
            stds.append(round(x.std().item(), 2))

        return stds


