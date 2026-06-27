# Methodology

How and why MonteCarloGPU works, from the financial math to the GPU design.

## Why Monte Carlo on a GPU

Monte Carlo pricing estimates the value of a derivative by simulating many
possible futures for the underlying asset and averaging the discounted payoff.
Each simulated path is independent of every other path, which makes the workload
embarrassingly parallel. That property is exactly what a GPU is built for.
Thousands of threads each walk one path at the same time, so the throughput
scales with the width of the device rather than the speed of a single core.
Banks run this kind of simulation continuously for pricing and risk, which is
why NVIDIA sells DGX and HGX systems into financial services.

The underlying model is geometric Brownian motion under the risk neutral
measure. A price step follows

    S(t + dt) = S(t) * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)

where r is the risk free rate, sigma is volatility, dt is the time step, and Z
is a standard normal draw. A European option cares only about the terminal price
S(T). An Asian option cares about the average price along the path. A barrier
option cares about whether the path ever crossed a level. Those three payoff
shapes drive the three kernels in this project.

## Random number generation

Every path needs its own stream of normal random numbers, and the quality of
those numbers matters. The CUDA kernels use cuRAND with the Philox generator.
Philox is counter based, which means each thread can seed its own independent
substream cheaply without a long warm up. It is fast, statistically strong, and
deterministic given a seed, so a run reproduces exactly. The European kernel
pulls four normals at a time with curand_normal4 to keep the RNG pipeline
saturated, since random number generation rather than arithmetic is often the
bottleneck in a Monte Carlo kernel.

Using the same seed across the Greek calculations is a deliberate variance
reduction technique known as common random numbers. When delta is estimated by
bumping the spot up and down and differencing, correlated paths make most of the
Monte Carlo noise cancel, so the finite difference estimate is far tighter than
two independent runs would give.

## Memory and computation tradeoffs

A European option never needs the full path stored, only the running log price,
so the kernel keeps a single scalar per thread and never touches global memory
until the reduction. An Asian option needs the path average, so it accumulates a
running sum as it steps. A barrier option needs to know if any step breached the
level, so it carries a boolean flag. None of the three ever writes a full path
to memory, which keeps the kernels compute bound rather than bandwidth bound.

Summing millions of per thread payoffs into one number is a reduction. The
kernels use a warp level reduction with __shfl_down_sync followed by a block
level reduction through a small shared memory buffer. This is the fastest
reduction pattern on modern NVIDIA architectures because the warp shuffle stays
in registers and needs no synchronization. Each block writes a single partial
sum, and the host adds the handful of block sums. Computation uses single
precision float because GPU single precision throughput is much higher than
double, and the statistical error of the estimate dominates the floating point
error at any realistic path count.

## Convergence and accuracy

Monte Carlo error shrinks as one over the square root of the number of paths.
To halve the error you need four times the paths. That sounds expensive, and on
a CPU it is, which is the whole motivation for the GPU. When one hundred million
paths take a fraction of a second the square root law stops being a wall. The
convergence study in this repo fits the slope of log standard error against log
path count and recovers a value of about negative one half, which is the
theoretical rate. The European price is validated directly against the closed
form Black Scholes solution, and the Monte Carlo estimate lands within a few
standard errors every time.

Further variance reduction is possible and is a natural extension. Antithetic
variates reuse each normal draw with its sign flipped. Control variates price a
related instrument with a known analytical value and correct the estimate by the
known error. Both reduce the paths needed for a target accuracy and would slot
into the same kernel structure.

## The neural surrogate

The ml directory trains a small neural network to approximate the pricing map
directly. The network takes the option parameters and returns a price in a
single forward pass, which prices a whole book in microseconds rather than
milliseconds. The tradeoff is a small bounded approximation error in exchange
for a very large latency reduction, which is the tradeoff a real time risk desk
makes. Training data is labeled with the exact Black Scholes price, so the model
learns a clean noise free target and its error is genuine approximation error.

## Relevance to NVIDIA financial services

This project is a compact version of a real workload. NVIDIA GPUs price
derivatives and compute risk for financial institutions every day. cuOPT
handles portfolio optimization, cuDF handles the data processing around it, and
the Monte Carlo pricing shown here is the simulation core. Packaging the whole
thing in a container that runs the same on a laptop, an HPC login node, and a
Kubernetes GPU pod is the part that makes it a workload rather than a script.
