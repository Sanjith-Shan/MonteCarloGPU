"""MonteCarloGPU Python package.

Reference CPU and PyTorch implementations of the Monte Carlo option pricing
engine. The C++/CUDA engine is the production path. This package is the
portable baseline that runs anywhere, drives the benchmark suite, generates the
charts, and feeds training data to the neural surrogate pricer in ml/.
"""

from .black_scholes import bs_price, bs_greeks, norm_cdf
from .cpu_pricer import price_european_cpu, price_asian_cpu, price_barrier_cpu
from .torch_pricer import TorchPricer, pick_device

__all__ = [
    "bs_price",
    "bs_greeks",
    "norm_cdf",
    "price_european_cpu",
    "price_asian_cpu",
    "price_barrier_cpu",
    "TorchPricer",
    "pick_device",
]

__version__ = "0.3.0"
