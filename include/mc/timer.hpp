#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <chrono>

namespace mc {

// Wall clock timer for host side measurements. Used for the CPU baseline and
// for end to end timing that includes host device transfers.
class HostTimer {
public:
    void start() { t0_ = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point t0_;
};

#ifdef __CUDACC__
// CUDA event based timer. Measures kernel execution on the device timeline,
// which is the fair way to report GPU compute time. Host timers include
// launch overhead and synchronization that we want to exclude.
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { cudaEventRecord(start_, 0); }
    float stop_ms() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
#endif

}  // namespace mc
