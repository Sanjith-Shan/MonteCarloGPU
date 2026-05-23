#include <benchmark/benchmark.h>

#include "mc/kernels.hpp"
#include "mc/types.hpp"

using namespace mc;

namespace {

OptionParams make_params(int paths) {
    OptionParams p;
    p.spot = 100.0f; p.strike = 100.0f; p.rate = 0.05f;
    p.vol = 0.20f; p.maturity = 1.0f; p.n_steps = 252;
    p.n_paths = paths; p.seed = 42;
    return p;
}

}  // namespace

// Sweep the European kernel across path counts. Google Benchmark reports the
// wall clock per iteration, which we convert to paths per second below.
static void BM_European(benchmark::State& state) {
    if (!cuda_available()) { state.SkipWithError("no CUDA device"); return; }
    OptionParams p = make_params(static_cast<int>(state.range(0)));
    p.type = OptionType::EuropeanCall;
    float se = 0.0f;
    for (auto _ : state) {
        float price = launch_european(p, &se);
        benchmark::DoNotOptimize(price);
    }
    state.SetItemsProcessed(state.iterations() * p.n_paths);
}
BENCHMARK(BM_European)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

static void BM_Asian(benchmark::State& state) {
    if (!cuda_available()) { state.SkipWithError("no CUDA device"); return; }
    OptionParams p = make_params(static_cast<int>(state.range(0)));
    p.type = OptionType::AsianCall;
    float se = 0.0f;
    for (auto _ : state) {
        float price = launch_asian(p, &se);
        benchmark::DoNotOptimize(price);
    }
    state.SetItemsProcessed(state.iterations() * p.n_paths);
}
BENCHMARK(BM_Asian)->Arg(1'000'000)->Arg(10'000'000)->Unit(benchmark::kMillisecond);

static void BM_Barrier(benchmark::State& state) {
    if (!cuda_available()) { state.SkipWithError("no CUDA device"); return; }
    OptionParams p = make_params(static_cast<int>(state.range(0)));
    p.type = OptionType::BarrierUpAndOutCall;
    p.barrier = 120.0f;
    float se = 0.0f;
    for (auto _ : state) {
        float price = launch_barrier(p, &se);
        benchmark::DoNotOptimize(price);
    }
    state.SetItemsProcessed(state.iterations() * p.n_paths);
}
BENCHMARK(BM_Barrier)->Arg(1'000'000)->Arg(10'000'000)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
