#pragma once

#include <string>

namespace mc {

// Which kind of derivative we are pricing.
enum class OptionType {
    EuropeanCall,
    EuropeanPut,
    AsianCall,
    BarrierUpAndOutCall
};

// Full description of a pricing problem. All rates and volatilities are
// annualized. Time is measured in years.
struct OptionParams {
    OptionType type = OptionType::EuropeanCall;
    float spot = 100.0f;      // S0, current underlying price
    float strike = 100.0f;    // K, strike price
    float rate = 0.05f;       // r, continuously compounded risk free rate
    float vol = 0.20f;        // sigma, annualized volatility
    float maturity = 1.0f;    // T, time to expiry in years
    float barrier = 0.0f;     // B, only used for barrier options
    int n_paths = 1000000;    // number of Monte Carlo paths
    int n_steps = 252;        // time steps per path (252 = trading days)
    unsigned long seed = 42;  // PRNG seed for reproducibility
};

// Risk sensitivities of the option value with respect to the inputs.
struct Greeks {
    float delta = 0.0f;  // dV/dS
    float gamma = 0.0f;  // d2V/dS2
    float vega = 0.0f;   // dV/dsigma
    float theta = 0.0f;  // dV/dT (per year)
};

// Result of a single pricing run.
struct PricingResult {
    float price = 0.0f;        // discounted expected payoff
    float std_error = 0.0f;    // standard error of the estimate
    float bs_price = 0.0f;     // Black Scholes analytical price (Europeans only)
    float abs_error = 0.0f;    // |price - bs_price|
    double elapsed_ms = 0.0;   // wall clock time for the GPU run
    long long paths = 0;       // number of paths actually simulated
};

inline std::string to_string(OptionType t) {
    switch (t) {
        case OptionType::EuropeanCall:        return "European Call";
        case OptionType::EuropeanPut:         return "European Put";
        case OptionType::AsianCall:           return "Asian Call";
        case OptionType::BarrierUpAndOutCall: return "Up-and-Out Barrier Call";
    }
    return "Unknown";
}

}  // namespace mc
