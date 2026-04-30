"""PyTorch Monte Carlo pricer.

A device agnostic tensor implementation that runs on CUDA, Apple MPS, or CPU
with the same code path. This mirrors the CUDA kernels in pure PyTorch so the
engine is portable to any accelerator PyTorch supports, and so the benchmark
can report a GPU number even on hardware without the CUDA toolkit installed.

Paths are simulated in batches so that pricing 100M paths never materializes
more than one batch of the price tensor at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Select the best available device. CUDA, then MPS, then CPU."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PriceResult:
    price: float
    std_error: float
    elapsed_ms: float
    device: str
    paths: int


def _sync(device: torch.device) -> None:
    # Force outstanding async kernels to finish so timing is honest.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class TorchPricer:
    """Batched Monte Carlo pricer backed by PyTorch tensors."""

    # Keep any single simulated tensor under this many elements so memory and
    # kernel launch timing stay stable across path counts and step counts.
    MAX_BATCH_ELEMENTS = 50_000_000

    def __init__(self, device: Optional[str] = None, batch_paths: int = 4_000_000,
                 dtype: torch.dtype = torch.float32):
        self.device = pick_device(device)
        self.batch_paths = batch_paths
        self.dtype = dtype

    def _effective_batch(self, n_steps: int) -> int:
        by_elements = max(1, self.MAX_BATCH_ELEMENTS // max(n_steps, 1))
        return min(self.batch_paths, by_elements)

    def _run(self, payoff_fn, n_paths, n_steps, r, T, seed):
        """Common batched driver. payoff_fn maps a (batch, steps) log return
        tensor to a (batch,) payoff tensor."""
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        disc = math.exp(-r * T)

        total = 0
        # Accumulate the reductions on the host in double precision. MPS has no
        # float64 support, so per batch sums come back as float32 scalars and we
        # widen them here. The statistical error dwarfs this rounding.
        payoff_sum = 0.0
        payoff_sq = 0.0

        batch = self._effective_batch(n_steps)

        _sync(self.device)
        import time
        t0 = time.perf_counter()

        while total < n_paths:
            m = min(batch, n_paths - total)
            z = torch.randn((m, n_steps), generator=gen, device=self.device,
                            dtype=self.dtype)
            payoff = payoff_fn(z)
            payoff_sum += payoff.sum().item()
            payoff_sq += (payoff * payoff).sum().item()
            total += m

        _sync(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mean = payoff_sum / n_paths
        var = max(payoff_sq / n_paths - mean * mean, 0.0)
        price = disc * mean
        std_error = disc * math.sqrt(var / n_paths)
        return PriceResult(price=price, std_error=std_error, elapsed_ms=elapsed_ms,
                           device=str(self.device), paths=n_paths)

    def price_european(self, S0, K, r, sigma, T, n_paths, n_steps=252, seed=42,
                       call=True) -> PriceResult:
        dt = T / n_steps
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)
        logS0 = math.log(S0)

        def payoff_fn(z):
            logST = logS0 + (drift + diffusion * z).sum(dim=1)
            ST = torch.exp(logST)
            if call:
                return torch.clamp(ST - K, min=0.0)
            return torch.clamp(K - ST, min=0.0)

        return self._run(payoff_fn, n_paths, n_steps, r, T, seed)

    def price_asian(self, S0, K, r, sigma, T, n_paths, n_steps=252, seed=42
                    ) -> PriceResult:
        dt = T / n_steps
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)
        logS0 = math.log(S0)

        def payoff_fn(z):
            log_paths = logS0 + torch.cumsum(drift + diffusion * z, dim=1)
            avg = torch.exp(log_paths).mean(dim=1)
            return torch.clamp(avg - K, min=0.0)

        return self._run(payoff_fn, n_paths, n_steps, r, T, seed)

    def price_barrier(self, S0, K, B, r, sigma, T, n_paths, n_steps=252, seed=42
                      ) -> PriceResult:
        dt = T / n_steps
        drift = (r - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)
        logS0 = math.log(S0)
        logB = math.log(B)

        def payoff_fn(z):
            log_paths = logS0 + torch.cumsum(drift + diffusion * z, dim=1)
            knocked = (log_paths >= logB).any(dim=1)
            ST = torch.exp(log_paths[:, -1])
            payoff = torch.clamp(ST - K, min=0.0)
            return torch.where(knocked, torch.zeros_like(payoff), payoff)

        return self._run(payoff_fn, n_paths, n_steps, r, T, seed)
