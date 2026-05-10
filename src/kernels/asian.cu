#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <vector>

#include "mc/kernels.hpp"
#include "reduction.cuh"

namespace mc {

namespace {

constexpr int kThreads = 256;

// Asian arithmetic average call. Unlike the European case the payoff depends
// on the average price over the whole path, so we accumulate a running sum as
// we step forward. The path itself is never stored, only the running sum.
__global__ void asian_kernel(float* block_sum,
                             float* block_sum_sq,
                             float S0, float K, float r, float sigma,
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
        float running = 0.0f;
        for (int step = 0; step < n_steps; ++step) {
            float z = curand_normal(&state);
            logS += drift + diffusion * z;
            running += expf(logS);
        }
        float avg = running / n_steps;
        float payoff = fmaxf(avg - K, 0.0f);
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

float launch_asian(const OptionParams& p, float* out_std_error) {
    int blocks = (p.n_paths + kThreads - 1) / kThreads;
    blocks = min(blocks, 4096);

    float* d_sum = nullptr;
    float* d_sum_sq = nullptr;
    cudaMalloc(&d_sum, blocks * sizeof(float));
    cudaMalloc(&d_sum_sq, blocks * sizeof(float));

    asian_kernel<<<blocks, kThreads>>>(d_sum, d_sum_sq, p.spot, p.strike,
                                       p.rate, p.vol, p.maturity, p.n_steps,
                                       p.n_paths, p.seed);

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

}  // namespace mc
