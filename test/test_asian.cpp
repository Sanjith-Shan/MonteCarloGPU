#include <gtest/gtest.h>

#include "mc/kernels.hpp"
#include "mc/pricer.hpp"

using namespace mc;

namespace {

OptionParams base_asian() {
    OptionParams p;
    p.type = OptionType::AsianCall;
    p.spot = 100.0f;
    p.strike = 100.0f;
    p.rate = 0.05f;
    p.vol = 0.20f;
    p.maturity = 1.0f;
    p.n_paths = 5'000'000;
    p.n_steps = 252;
    p.seed = 42;
    return p;
}

#define REQUIRE_CUDA() \
    if (!cuda_available()) GTEST_SKIP() << "no CUDA device"

}  // namespace

// An arithmetic average Asian call is always cheaper than the otherwise
// identical European call because averaging dampens the terminal variance.
TEST(Asian, CheaperThanEuropean) {
    REQUIRE_CUDA();
    OptionParams asian = base_asian();
    OptionParams euro = asian;
    euro.type = OptionType::EuropeanCall;

    Pricer pricer;
    float asian_price = pricer.price(asian).price;
    float euro_price = pricer.price(euro).price;
    EXPECT_GT(asian_price, 0.0f);
    EXPECT_LT(asian_price, euro_price);
}

// Reference value from a fine grained independent run. Kept loose so it is a
// regression guard rather than a brittle exact match.
TEST(Asian, NearKnownReference) {
    REQUIRE_CUDA();
    OptionParams p = base_asian();
    Pricer pricer;
    float price = pricer.price(p).price;
    EXPECT_NEAR(price, 5.77f, 0.15f);
}
