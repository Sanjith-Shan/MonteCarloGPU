"""Neural surrogate pricer model.

A small fully connected network that maps option parameters to a price. Once
trained it prices a whole batch of options in a single forward pass, which is
orders of magnitude faster than running Monte Carlo per option. Desks use
surrogates like this for real time risk where approximate prices at very low
latency beat exact prices that arrive too late.

The standardization statistics are stored inside the module so a saved
checkpoint is fully self contained and inference needs no external state.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SurrogatePricer(nn.Module):
    def __init__(self, in_features: int = 5, hidden_sizes: List[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128, 128]

        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Registered as buffers so they move with .to(device) and save with the
        # state dict. Filled in by set_standardizer before training.
        self.register_buffer("feat_mean", torch.zeros(in_features))
        self.register_buffer("feat_std", torch.ones(in_features))

    def set_standardizer(self, mean, std):
        self.feat_mean.copy_(torch.as_tensor(mean))
        self.feat_std.copy_(torch.as_tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feat_mean) / self.feat_std
        # A call price is never negative, so softplus keeps the output in range
        # and helps the deep out of the money tail behave.
        return torch.nn.functional.softplus(self.net(x))


def build_model(model_cfg, in_features: int = 5) -> SurrogatePricer:
    return SurrogatePricer(
        in_features=in_features,
        hidden_sizes=model_cfg.get("hidden_sizes", [128, 128, 128]),
        dropout=model_cfg.get("dropout", 0.0),
    )
