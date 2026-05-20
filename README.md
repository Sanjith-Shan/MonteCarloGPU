# MonteCarloGPU

GPU accelerated Monte Carlo option pricing engine.

Prices European, Asian, and barrier options with CUDA accelerated Monte Carlo
simulation and validates them against the analytical Black-Scholes solution. A
portable PyTorch path runs the same pricing on CUDA, Apple MPS, or CPU.

## Quick start

```bash
pip install -r python/requirements.txt
python3 python/benchmark.py
python3 -m pytest
```

## Building the CUDA engine

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/montecarlo_gpu --type european --paths 10000000 --greeks
```

More to come.
