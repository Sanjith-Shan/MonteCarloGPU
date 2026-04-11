#include "mc/black_scholes.hpp"

#include <cmath>

namespace mc {

double norm_cdf(double x) {
    // 0.5 * erfc(-x / sqrt(2)) is the standard normal CDF and is numerically
    // stable in both tails.
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

namespace {

struct D12 {
    double d1;
    double d2;
};

D12 compute_d12(const OptionParams& p) {
    const double S = p.spot, K = p.strike, r = p.rate;
    const double sig = p.vol, T = p.maturity;
    const double d1 = (std::log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * std::sqrt(T));
    const double d2 = d1 - sig * std::sqrt(T);
    return {d1, d2};
}

}  // namespace

double black_scholes_price(const OptionParams& p) {
    const auto [d1, d2] = compute_d12(p);
    const double S = p.spot, K = p.strike, r = p.rate, T = p.maturity;
    const double disc = std::exp(-r * T);

    if (p.type == OptionType::EuropeanCall) {
        return S * norm_cdf(d1) - K * disc * norm_cdf(d2);
    }
    // Put via put call parity encoded directly.
    return K * disc * norm_cdf(-d2) - S * norm_cdf(-d1);
}

Greeks black_scholes_greeks(const OptionParams& p) {
    const auto [d1, d2] = compute_d12(p);
    const double S = p.spot, K = p.strike, r = p.rate;
    const double sig = p.vol, T = p.maturity;
    const double disc = std::exp(-r * T);
    const double sqrtT = std::sqrt(T);
    const bool is_call = (p.type == OptionType::EuropeanCall);

    Greeks g;
    g.delta = static_cast<float>(is_call ? norm_cdf(d1) : norm_cdf(d1) - 1.0);
    g.gamma = static_cast<float>(norm_pdf(d1) / (S * sig * sqrtT));
    g.vega = static_cast<float>(S * norm_pdf(d1) * sqrtT);
    // Theta reported per year. Divide by 365 for a per calendar day figure.
    const double term1 = -(S * norm_pdf(d1) * sig) / (2.0 * sqrtT);
    if (is_call) {
        g.theta = static_cast<float>(term1 - r * K * disc * norm_cdf(d2));
    } else {
        g.theta = static_cast<float>(term1 + r * K * disc * norm_cdf(-d2));
    }
    return g;
}

}  // namespace mc
