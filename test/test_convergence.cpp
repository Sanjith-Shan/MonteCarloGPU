#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "mc/kernels.hpp"
#include "mc/pricer.hpp"

using namespace mc;

namespace {

#define REQUIRE_CUDA() \
    if (!cuda_available()) GTEST_SKIP() << "no CUDA device"

}  // namespace

// Monte Carlo error scales as 1/sqrt(N). If we fit log(std_error) against
// log(N) the slope should be close to -0.5. This test locks in that theory.
TEST(Convergence, StandardErrorScalesAsInverseSqrtN) {
    REQUIRE_CUDA();
    OptionParams p;
    p.type = OptionType::EuropeanCall;
    p.spot = 100.0f; p.strike = 100.0f; p.rate = 0.05f;
    p.vol = 0.20f; p.maturity = 1.0f; p.n_steps = 100; p.seed = 42;

    std::vector<int> counts = {100'000, 400'000, 1'600'000, 6'400'000};
    std::vector<double> log_n, log_se;

    Pricer pricer;
    for (int n : counts) {
        p.n_paths = n;
        PricingResult r = pricer.price(p);
        log_n.push_back(std::log(static_cast<double>(n)));
        log_se.push_back(std::log(r.std_error));
    }

    // Ordinary least squares slope of log_se vs log_n.
    double mean_x = 0, mean_y = 0;
    for (size_t i = 0; i < log_n.size(); ++i) { mean_x += log_n[i]; mean_y += log_se[i]; }
    mean_x /= log_n.size();
    mean_y /= log_se.size();

    double num = 0, den = 0;
    for (size_t i = 0; i < log_n.size(); ++i) {
        num += (log_n[i] - mean_x) * (log_se[i] - mean_y);
        den += (log_n[i] - mean_x) * (log_n[i] - mean_x);
    }
    double slope = num / den;
    EXPECT_NEAR(slope, -0.5, 0.08);
}
