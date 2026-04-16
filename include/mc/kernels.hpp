#pragma once

#include "mc/types.hpp"

namespace mc {

// Host side launchers for the CUDA Monte Carlo kernels. Each launcher handles
// grid configuration, device memory, the block level reduction, and the final
// discount. They return the discounted price estimate and write the standard
// error through the out parameter.
//
// These declarations are compiled by both host (.cpp) and device (.cu) code,
// so they carry no CUDA specific types in the signature.

// European call or put. Only the terminal price matters, so paths are not
// stored. This is the fastest kernel and the headline benchmark.
float launch_european(const OptionParams& p, float* out_std_error);

// Asian arithmetic average call. Tracks a running sum of the price across all
// time steps and prices off the path average.
float launch_asian(const OptionParams& p, float* out_std_error);

// Up-and-out barrier call. The option knocks out (pays zero) if the path ever
// crosses the barrier. Requires inspecting every step of every path.
float launch_barrier(const OptionParams& p, float* out_std_error);

// Returns true if a CUDA capable device is visible at runtime. The CLI and
// tests use this to fall back to the CPU baseline gracefully.
bool cuda_available();

}  // namespace mc
