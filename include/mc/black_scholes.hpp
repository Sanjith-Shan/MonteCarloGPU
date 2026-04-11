#pragma once

#include "mc/types.hpp"

namespace mc {

// Analytical Black Scholes pricing for European options. These closed form
// solutions are the ground truth we validate the Monte Carlo estimates
// against. Only Europeans have a closed form, so Asian and barrier options
// fall back to Monte Carlo with no analytical reference.

// Standard normal cumulative distribution function.
double norm_cdf(double x);

// Standard normal probability density function.
double norm_pdf(double x);

// Price a European call or put analytically.
double black_scholes_price(const OptionParams& p);

// Analytical Greeks for a European option. Used to validate the finite
// difference Greeks computed on the GPU.
Greeks black_scholes_greeks(const OptionParams& p);

}  // namespace mc
