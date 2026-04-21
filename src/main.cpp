#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mc/black_scholes.hpp"
#include "mc/kernels.hpp"
#include "mc/pricer.hpp"
#include "mc/types.hpp"

using namespace mc;

namespace {

void print_usage() {
    std::printf(
        "MonteCarloGPU - GPU accelerated Monte Carlo option pricer\n\n"
        "Usage: montecarlo_gpu [options]\n\n"
        "  --type      european|put|asian|barrier   (default european)\n"
        "  --spot      initial underlying price      (default 100)\n"
        "  --strike    strike price                  (default 100)\n"
        "  --rate      risk free rate                (default 0.05)\n"
        "  --vol       volatility                    (default 0.20)\n"
        "  --maturity  years to expiry               (default 1.0)\n"
        "  --barrier   knock out level (barrier only)(default 120)\n"
        "  --paths     number of Monte Carlo paths   (default 10000000)\n"
        "  --steps     time steps per path           (default 252)\n"
        "  --seed      PRNG seed                      (default 42)\n"
        "  --greeks    also compute Greeks\n"
        "  --output    text|json                     (default text)\n"
        "  --help      show this message\n");
}

OptionType parse_type(const std::string& s) {
    if (s == "european" || s == "call") return OptionType::EuropeanCall;
    if (s == "put") return OptionType::EuropeanPut;
    if (s == "asian") return OptionType::AsianCall;
    if (s == "barrier") return OptionType::BarrierUpAndOutCall;
    std::fprintf(stderr, "unknown option type '%s'\n", s.c_str());
    std::exit(1);
}

// Minimal flag parser. Good enough for a benchmarking CLI and keeps the
// dependency footprint at zero.
const char* arg_value(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return nullptr;
}

bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return true;
    }
    return false;
}

void print_text(const OptionParams& p, const PricingResult& r,
                bool want_greeks, const Greeks* g) {
    std::printf("Option Type:     %s\n", to_string(p.type).c_str());
    std::printf("Spot:            %.2f\n", p.spot);
    std::printf("Strike:          %.2f\n", p.strike);
    std::printf("Risk-free Rate:  %.2f%%\n", p.rate * 100.0f);
    std::printf("Volatility:      %.2f%%\n", p.vol * 100.0f);
    std::printf("Maturity:        %.2f years\n", p.maturity);
    std::printf("Paths:           %lld\n", r.paths);
    std::printf("Steps:           %d\n\n", p.n_steps);

    std::printf("Monte Carlo Price:   %.4f\n", r.price);
    if (r.bs_price > 0.0f) {
        std::printf("Black-Scholes Price: %.4f\n", r.bs_price);
        std::printf("Absolute Error:      %.4f\n", r.abs_error);
    }
    std::printf("Standard Error:      %.4f\n", r.std_error);
    std::printf("GPU Time:            %.1f ms\n", r.elapsed_ms);

    if (want_greeks && g) {
        std::printf("\nGreeks (finite difference):\n");
        std::printf("  Delta:  %.4f\n", g->delta);
        std::printf("  Gamma:  %.4f\n", g->gamma);
        std::printf("  Vega:   %.4f\n", g->vega);
        std::printf("  Theta:  %.4f\n", g->theta);
    }
}

void print_json(const OptionParams& p, const PricingResult& r,
                bool want_greeks, const Greeks* g) {
    std::printf("{\n");
    std::printf("  \"type\": \"%s\",\n", to_string(p.type).c_str());
    std::printf("  \"spot\": %.4f,\n", p.spot);
    std::printf("  \"strike\": %.4f,\n", p.strike);
    std::printf("  \"paths\": %lld,\n", r.paths);
    std::printf("  \"mc_price\": %.6f,\n", r.price);
    std::printf("  \"bs_price\": %.6f,\n", r.bs_price);
    std::printf("  \"abs_error\": %.6f,\n", r.abs_error);
    std::printf("  \"std_error\": %.6f,\n", r.std_error);
    std::printf("  \"gpu_ms\": %.3f", r.elapsed_ms);
    if (want_greeks && g) {
        std::printf(",\n  \"greeks\": {\"delta\": %.6f, \"gamma\": %.6f, "
                    "\"vega\": %.6f, \"theta\": %.6f}\n", g->delta, g->gamma,
                    g->vega, g->theta);
    } else {
        std::printf("\n");
    }
    std::printf("}\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help")) {
        print_usage();
        return 0;
    }

    OptionParams p;
    if (const char* v = arg_value(argc, argv, "--type"))     p.type = parse_type(v);
    if (const char* v = arg_value(argc, argv, "--spot"))     p.spot = std::atof(v);
    if (const char* v = arg_value(argc, argv, "--strike"))   p.strike = std::atof(v);
    if (const char* v = arg_value(argc, argv, "--rate"))     p.rate = std::atof(v);
    if (const char* v = arg_value(argc, argv, "--vol"))      p.vol = std::atof(v);
    if (const char* v = arg_value(argc, argv, "--maturity")) p.maturity = std::atof(v);
    if (const char* v = arg_value(argc, argv, "--barrier"))  p.barrier = std::atof(v);
    if (const char* v = arg_value(argc, argv, "--paths"))    p.n_paths = std::atoi(v);
    if (const char* v = arg_value(argc, argv, "--steps"))    p.n_steps = std::atoi(v);
    if (const char* v = arg_value(argc, argv, "--seed"))     p.seed = std::strtoul(v, nullptr, 10);

    if (p.type == OptionType::BarrierUpAndOutCall && p.barrier <= 0.0f) {
        p.barrier = 120.0f;
    }

    const bool want_greeks = has_flag(argc, argv, "--greeks");
    const char* output = arg_value(argc, argv, "--output");
    const bool json = output && std::strcmp(output, "json") == 0;

    if (!cuda_available()) {
        std::fprintf(stderr,
                     "No CUDA device detected. Run inside the GPU container or "
                     "use the Python CPU baseline (python/benchmark.py).\n");
        return 2;
    }

    Pricer pricer;
    PricingResult res = pricer.price(p);

    Greeks g;
    if (want_greeks) g = pricer.compute_greeks(p);

    if (json) {
        print_json(p, res, want_greeks, want_greeks ? &g : nullptr);
    } else {
        print_text(p, res, want_greeks, want_greeks ? &g : nullptr);
    }
    return 0;
}
