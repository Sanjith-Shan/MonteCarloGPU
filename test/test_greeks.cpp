#include <gtest/gtest.h>

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

#define REQUIRE_CUDA() \
    if (!cuda_available()) GTEST_SKIP() << "no CUDA device"

}  // namespace

TEST(Greeks, DeltaInUnitInterval) {
    REQUIRE_CUDA();
    Pricer pricer;
    Greeks g = pricer.compute_greeks(base_call());
    EXPECT_GE(g.delta, 0.0f);
    EXPECT_LE(g.delta, 1.0f);
}

TEST(Greeks, GammaAndVegaPositive) {
    REQUIRE_CUDA();
    Pricer pricer;
    Greeks g = pricer.compute_greeks(base_call());
    EXPECT_GT(g.gamma, 0.0f);
    EXPECT_GT(g.vega, 0.0f);
}

TEST(Greeks, MatchAnalyticalWithinTolerance) {
    REQUIRE_CUDA();
    OptionParams p = base_call();
    Pricer pricer;
    Greeks mc = pricer.compute_greeks(p);
    Greeks bs = black_scholes_greeks(p);

    // Finite difference Greeks with common random numbers should land close to
    // the analytical values. Vega and theta carry the loosest tolerance.
    EXPECT_NEAR(mc.delta, bs.delta, 0.01f);
    EXPECT_NEAR(mc.gamma, bs.gamma, 0.01f);
    EXPECT_NEAR(mc.vega, bs.vega, 1.0f);
    EXPECT_NEAR(mc.theta, bs.theta, 1.0f);
}
