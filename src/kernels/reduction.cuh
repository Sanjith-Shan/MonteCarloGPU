#pragma once

#include <cuda_runtime.h>

namespace mc {

// Warp level reduction using shuffle instructions. Sums a value across the 32
// lanes of a warp with no shared memory and no synchronization. This is the
// fastest reduction primitive on modern NVIDIA architectures.
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block level reduction. Each warp reduces its own lanes, the warp partials go
// through shared memory, and the first warp reduces those partials. Returns
// the block sum in thread 0 (garbage in the other threads).
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];  // one slot per warp, 32 warps max
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Only the first warp participates in the final reduction.
    val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

}  // namespace mc
