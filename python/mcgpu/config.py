"""Shared benchmark configuration.

Central place for the reference option and the path count sweep so the
benchmark, convergence, and plotting scripts all agree on the same setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RefOption:
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    vol: float = 0.20
    maturity: float = 1.0
    barrier: float = 120.0
    n_steps: int = 252
    seed: int = 42


# Path counts for the scaling sweep. Trimmed automatically by the benchmark
# when running on a slow CPU only host so a smoke run stays quick.
DEFAULT_PATH_SWEEP: List[int] = [
    10_000, 50_000, 100_000, 500_000, 1_000_000,
    5_000_000, 10_000_000, 50_000_000,
]

REFERENCE = RefOption()
