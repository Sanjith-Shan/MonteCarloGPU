#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <vector>

#include "mc/kernels.hpp"
#include "reduction.cuh"

namespace mc {

namespace {

constexpr int kThreads = 256;

// One thread simulates one full path of geometric Brownian motion and returns
// the discounted payoff of a European option. The payoffs are reduced to a
// single sum per block, plus a sum of squares so the host can form the
// standard error.
__global__ void european_kernel(float* block_sum,
                                 float* block_sum_sq,
                                 float S0, float K, float r, float sigma,
                                 float T, int n_steps, int n_paths,
                                 unsigned long seed, bool is_call) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Philox is counter based, cheap to seed per thread, and statistically
    // solid. Each thread gets an independent substream via its global id.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, gid, 0, &state);

    const float dt = T / n_steps;
    const float drift = (r - 0.5f * sigma * sigma) * dt;
    const float diffusion = sigma * sqrtf(dt);

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Grid stride loop so any launch configuration covers all paths.
    for (int p = gid; p < n_paths; p += gridDim.x * blockDim.x) {
        float logS = logf(S0);
        // Draw normals four at a time to keep the RNG pipeline saturated.
        int step = 0;
        while (step < n_steps) {
            float4 z = curand_normal4(&state);
            logS += drift + diffusion * z.x; if (++step >= n_steps) break;
            logS += drift + diffusion * z.y; if (++step >= n_steps) break;
            logS += drift + diffusion * z.z; if (++step >= n_steps) break;
            logS += drift + diffusion * z.w; ++step;
        }
        float ST = expf(logS);
        float payoff = is_call ? fmaxf(ST - K, 0.0f) : fmaxf(K - ST, 0.0f);
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

float launch_european(const OptionParams& p, float* out_std_error) {
    const bool is_call = (p.type == OptionType::EuropeanCall);

    // Cap the grid at a few thousand blocks and rely on the grid stride loop
    // to cover very large path counts without oversubscribing the device.
    int blocks = (p.n_paths + kThreads - 1) / kThreads;
    blocks = min(blocks, 4096);

    float* d_sum = nullptr;
    float* d_sum_sq = nullptr;
    cudaMalloc(&d_sum, blocks * sizeof(float));
    cudaMalloc(&d_sum_sq, blocks * sizeof(float));

    european_kernel<<<blocks, kThreads>>>(d_sum, d_sum_sq, p.spot, p.strike,
                                           p.rate, p.vol, p.maturity, p.n_steps,
                                           p.n_paths, p.seed, is_call);

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

    const double price = discount * mean_payoff;
    if (out_std_error) {
        *out_std_error = static_cast<float>(discount * std::sqrt(var_payoff / n));
    }
    return static_cast<float>(price);
}

}  // namespace mc
