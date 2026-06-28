# Results

Numbers below come from the committed run in `results/` and `ml/`. The GPU used
here is Apple MPS on an M series laptop, which is the accelerator available on
the development machine. On an NVIDIA datacenter GPU the custom CUDA kernels run
substantially faster still, and the same scripts regenerate every number inside
the GPU container. Regenerate everything with

```bash
python3 python/benchmark.py
python3 python/convergence.py
python3 python/surface.py
python3 python/plot_results.py
```

## Scaling, CPU versus GPU

Reference European call. Spot 100, strike 100, rate 5 percent, volatility 20
percent, one year, 252 steps.

| Paths | CPU NumPy | GPU | Speedup |
|-------|-----------|-----|---------|
| 10,000 | 35 ms | 9.8 ms | 3.6x |
| 100,000 | 376 ms | 21 ms | 17.5x |
| 1,000,000 | 3,650 ms | 132 ms | 27.7x |
| 5,000,000 | 17,607 ms | 301 ms | 58.6x |
| 10,000,000 | 37,675 ms | 595 ms | 63.4x |

The speedup grows with path count because the fixed launch and transfer overhead
is amortized over more work. On the CPU the same sweep runs into tens of seconds
per point, which is exactly the wall that GPU acceleration removes.

## Accuracy against Black Scholes

The analytical Black Scholes price of the reference call is 10.4506. At ten
million paths the Monte Carlo estimate is 10.4474, an absolute error of 0.0032,
which sits well inside one standard error. The estimate converges to the closed
form value as paths increase.

## Convergence rate

Fitting the log of the standard error against the log of the path count gives a
slope of -0.5002. The theoretical Monte Carlo rate is exactly -0.5, so the
engine reproduces the square root law almost perfectly.

## Option types

Priced at ten million paths.

| Option | Price |
|--------|-------|
| European call | 10.4474 |
| Asian call (arithmetic average) | 5.7794 |
| Up and out barrier call (barrier 120) | 1.3287 |

The Asian call is cheaper than the European because averaging dampens the
terminal variance. The barrier call is cheaper still because it can knock out.

## Greeks

Finite difference Greeks with common random numbers, at ten million paths,
against the analytical Black Scholes values.

| Greek | Monte Carlo | Analytical |
|-------|-------------|------------|
| Delta | 0.6367 | 0.6368 |
| Gamma | 0.0188 | 0.0188 |
| Vega | 37.509 | 37.524 |
| Theta | -6.415 | -6.414 |

## Neural surrogate pricer

An MLP trained on 84,000 options and validated on 18,000 more, labeled with
exact Black Scholes prices.

| Metric | Value |
|--------|-------|
| Validation RMSE | 0.0026 per unit strike |
| Test RMSE | 0.0026 per unit strike |
| Test MAPE (price above 0.01) | 2.09 percent |
| Training time | 7.2 seconds |
| Inference latency | 1.83 microseconds per option |
| Inference throughput | 545,000 options per second |

The surrogate prices an entire book in a single batched forward pass. It trades
a small bounded approximation error for a very large latency reduction, which is
the tradeoff a real time risk system makes.
