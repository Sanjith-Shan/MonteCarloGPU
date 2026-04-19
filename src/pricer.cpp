#include "mc/pricer.hpp"

#include <cmath>
#include <stdexcept>

#include "mc/black_scholes.hpp"
#include "mc/kernels.hpp"
#include "mc/timer.hpp"

namespace mc {

namespace {

float dispatch_kernel(const OptionParams& p, float* std_error) {
    switch (p.type) {
        case OptionType::EuropeanCall:
        case OptionType::EuropeanPut:
            return launch_european(p, std_error);
        case OptionType::AsianCall:
            return launch_asian(p, std_error);
        case OptionType::BarrierUpAndOutCall:
            return launch_barrier(p, std_error);
    }
    throw std::runtime_error("unknown option type");
}

bool has_analytical(OptionType t) {
    return t == OptionType::EuropeanCall || t == OptionType::EuropeanPut;
}

}  // namespace

PricingResult Pricer::price(const OptionParams& p) const {
    if (!cuda_available()) {
        throw std::runtime_error("no CUDA device available");
    }

    PricingResult res;
    res.paths = p.n_paths;

    HostTimer timer;
    timer.start();
    float std_err = 0.0f;
    res.price = dispatch_kernel(p, &std_err);
    res.elapsed_ms = timer.stop_ms();
    res.std_error = std_err;

    if (has_analytical(p.type)) {
        res.bs_price = static_cast<float>(black_scholes_price(p));
        res.abs_error = std::fabs(res.price - res.bs_price);
    }
    return res;
}

Greeks Pricer::compute_greeks(const OptionParams& p) const {
    // Central finite differences with common random numbers. Every bumped run
    // reuses the same seed so the paths are correlated and most of the Monte
    // Carlo variance cancels between the up and down estimates.
    auto price_with = [&](float spot, float vol, float mat) {
        OptionParams q = p;
        q.spot = spot;
        q.vol = vol;
        q.maturity = mat;
        float se = 0.0f;
        return dispatch_kernel(q, &se);
    };

    const float hS = 0.01f * p.spot;
    const float hV = 0.001f;
    const float hT = 1.0f / 365.0f;

    const float base = price_with(p.spot, p.vol, p.maturity);
    const float up = price_with(p.spot + hS, p.vol, p.maturity);
    const float down = price_with(p.spot - hS, p.vol, p.maturity);
    const float vega_up = price_with(p.spot, p.vol + hV, p.maturity);
    const float vega_down = price_with(p.spot, p.vol - hV, p.maturity);
    const float theta_fwd = price_with(p.spot, p.vol, p.maturity - hT);

    Greeks g;
    g.delta = (up - down) / (2.0f * hS);
    g.gamma = (up - 2.0f * base + down) / (hS * hS);
    g.vega = (vega_up - vega_down) / (2.0f * hV);
    // Theta is the sensitivity to the passage of time, hence the negative.
    g.theta = (theta_fwd - base) / hT;
    return g;
}

}  // namespace mc
