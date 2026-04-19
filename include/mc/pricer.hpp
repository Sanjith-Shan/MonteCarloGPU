#pragma once

#include "mc/types.hpp"

namespace mc {

// High level pricing interface. This is the entry point the CLI and tests use.
// It dispatches to the correct CUDA kernel, fills in the Black Scholes
// reference for European options, and records timing.
class Pricer {
public:
    // Price a single option. Runs on the GPU if one is available, otherwise
    // throws. Callers that want a CPU fallback should check cuda_available().
    PricingResult price(const OptionParams& p) const;

    // Compute the four first and second order Greeks by finite difference.
    // Each Greek reruns the Monte Carlo with a bumped parameter using the same
    // seed so that common random numbers cancel most of the variance.
    Greeks compute_greeks(const OptionParams& p) const;
};

}  // namespace mc
