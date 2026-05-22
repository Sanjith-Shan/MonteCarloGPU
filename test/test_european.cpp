#include <gtest/gtest.h>

#include <cmath>

#include "mc/black_scholes.hpp"
#include "mc/kernels.hpp"
#include "mc/pricer.hpp"

using namespace mc;

namespace {

OptionParams base_call() {
    OptionParams p;
    p.type = OptionType::EuropeanCall;
    p.spot = 100.0f;
    p.strike = 100.0f;
    p.rate = 0.05f;
    p.vol = 0.20f;
    p.maturity = 1.0f;
    p.n_paths = 10'000'000;
    p.n_steps = 252;
    p.seed = 42;
    return p;
}

// Every GPU test is skipped cleanly when no device is present so the suite is
// still meaningful on a CPU only CI runner.
#define REQUIRE_CUDA() \
    if (!cuda_available()) GTEST_SKIP() << "no CUDA device"

}  // namespace

TEST(European, MatchesBlackScholesWithinThreeStdErrors) {
    REQUIRE_CUDA();
    OptionParams p = base_call();
    Pricer pricer;
    PricingResult r = pricer.price(p);
    double bs = black_scholes_price(p);
    // Three standard errors is a 99.7% confidence band.
    EXPECT_NEAR(r.price, bs, 3.0f * r.std_error);
}

TEST(European, PutCallParityHolds) {
    REQUIRE_CUDA();
    OptionParams call = base_call();
    OptionParams put = call;
    put.type = OptionType::EuropeanPut;

    Pricer pricer;
    float c = pricer.price(call).price;
    float pv = pricer.price(put).price;

    // C - P = S - K exp(-rT)
    float parity = call.spot - call.strike * std::exp(-call.rate * call.maturity);
    EXPECT_NEAR(c - pv, parity, 0.05f);
}

TEST(European, StandardErrorIsPositiveAndSmall) {
    REQUIRE_CUDA();
    OptionParams p = base_call();
    Pricer pricer;
    PricingResult r = pricer.price(p);
    EXPECT_GT(r.std_error, 0.0f);
    EXPECT_LT(r.std_error, 0.05f);  // 10M paths should be tight
}
