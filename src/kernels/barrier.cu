#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <vector>

#include "mc/kernels.hpp"
#include "reduction.cuh"

namespace mc {

namespace {

constexpr int kThreads = 256;

// Up-and-out barrier call. The option behaves like a European call unless the
// path ever touches or crosses the barrier B, in which case it knocks out and
// pays nothing. This forces us to inspect every step rather than just the
// terminal value, which is why barrier options are more expensive than
// Europeans on the same path count.
__global__ void barrier_kernel(float* block_sum,
                              float* block_sum_sq,
                              float S0, float K, float B, float r, float sigma,
                              float T, int n_steps, int n_paths,
                              unsigned long seed) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, gid, 0, &state);

    const float dt = T / n_steps;
    const float drift = (r - 0.5f * sigma * sigma) * dt;
    const float diffusion = sigma * sqrtf(dt);

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int p = gid; p < n_paths; p += gridDim.x * blockDim.x) {
        float logS = logf(S0);
        bool knocked_out = false;
        const float logB = logf(B);
        for (int step = 0; step < n_steps; ++step) {
            float z = curand_normal(&state);
            logS += drift + diffusion * z;
            if (logS >= logB) { knocked_out = true; break; }
        }
        float payoff = 0.0f;
        if (!knocked_out) {
            float ST = expf(logS);
            payoff = fmaxf(ST - K, 0.0f);
        }
        local_sum += payoff;
        local_sum_sq += payoff * payoff;
    }

    float bsum = block_reduce_sum(local_sum);
    float bsum_sq = block_reduce_sum(local_sum_sq);
    if (threadIdx.x == 0) {
        block_sum[blockIdx.x] = bsum;
        block_sum_sq[blockIdx.x] = bsum_sq;
    }
}

}  // namespace

float launch_barrier(const OptionParams& p, float* out_std_error) {
    int blocks = (p.n_paths + kThreads - 1) / kThreads;
    blocks = min(blocks, 4096);

    float* d_sum = nullptr;
    float* d_sum_sq = nullptr;
    cudaMalloc(&d_sum, blocks * sizeof(float));
    cudaMalloc(&d_sum_sq, blocks * sizeof(float));

    barrier_kernel<<<blocks, kThreads>>>(d_sum, d_sum_sq, p.spot, p.strike,
                                         p.barrier, p.rate, p.vol, p.maturity,
                                         p.n_steps, p.n_paths, p.seed);

    std::vector<float> h_sum(blocks), h_sum_sq(blocks);
    cudaMemcpy(h_sum.data(), d_sum, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum_sq.data(), d_sum_sq, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    cudaFree(d_sum_sq);

    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < blocks; ++i) {
        sum += h_sum[i];
        sum_sq += h_sum_sq[i];
    }

    const double n = static_cast<double>(p.n_paths);
    const double discount = std::exp(-p.rate * p.maturity);
    const double mean_payoff = sum / n;
    const double var_payoff = sum_sq / n - mean_payoff * mean_payoff;

    if (out_std_error) {
        *out_std_error = static_cast<float>(discount * std::sqrt(var_payoff / n));
    }
    return static_cast<float>(discount * mean_payoff);
}

// Runtime probe for a usable CUDA device. Lives here so a single translation
// unit owns the query. The CLI and tests use it to fall back to CPU.
bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

}  // namespace mc
