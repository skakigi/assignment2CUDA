#pragma once

// Fixed-template SumCheck evaluation strategy helpers.
// Included inside the same namespace as the original definitions.

constexpr int kMleItemsPerThread = 4;

enum class SpecEvalStrategy {
    kBaselineShared = 0,
    kMleTiledShared = 1,
    kMleTiledWarp = 2,
    kMleTiledReduceIntrinsics = 3,
    kCompiledTemplate
};

SpecEvalStrategy select_spec_eval_strategy() {
    const char* env = std::getenv("SC_EVAL_VARIANT");
    if (env == nullptr) {
        return SpecEvalStrategy::kBaselineShared;
    }
    const std::string value(env);
    if (value == "baseline" || value == "shared") {
        return SpecEvalStrategy::kBaselineShared;
    }
    if (value == "mle_tiled" || value == "mle_tiled_shared" || value == "tile") {
        return SpecEvalStrategy::kMleTiledShared;
    }
    if (value == "mle_tiled_warp" || value == "warp" || value == "shuffle") {
        return SpecEvalStrategy::kMleTiledWarp;
    }
    if (value == "mle_tiled_reduce" || value == "reduce" || value == "intrinsics") {
        return SpecEvalStrategy::kMleTiledReduceIntrinsics;
    }
    
    if (value == "compiled" || value == "compile" ||
        value == "templated" || value == "template") {
        return SpecEvalStrategy::kCompiledTemplate;
    }
    return SpecEvalStrategy::kBaselineShared;
}

inline bool supports_compiled_template_for_poly(int poly_id) {
    // Compiled-template kernels are enabled one polynomial at a time.
    // This intentionally returns false until a compiled kernel is added.
    (void)poly_id;
    return false;
}

inline SpecEvalStrategy resolve_spec_eval_strategy_for_launch(
    SpecEvalStrategy selected,
    int poly_id) {
    // `compiled` is now a first-class strategy, but it only routes to
    // compiled kernels for templates explicitly marked as supported.
    if (selected == SpecEvalStrategy::kCompiledTemplate &&
        !supports_compiled_template_for_poly(poly_id)) {
        return SpecEvalStrategy::kBaselineShared;
    }
    return selected;
}

inline SpecEvalStrategy resolve_spec_eval_strategy_for_launch_no_poly(
    SpecEvalStrategy selected) {
    // Generic/non-fixed-template paths do not have a fixed poly_id.
    // Treat compiled as baseline until compiled support is only used by
    // fixed-template dispatch.
    if (selected == SpecEvalStrategy::kCompiledTemplate) {
        return SpecEvalStrategy::kBaselineShared;
    }
    return selected;
}

inline int eval_threads_for_strategy(SpecEvalStrategy variant) {
    return (variant == SpecEvalStrategy::kBaselineShared) ? 256 : 128;
}

inline uint64_t eval_items_per_block_for_strategy(SpecEvalStrategy variant, int eval_threads) {
    if (variant == SpecEvalStrategy::kBaselineShared) {
        return static_cast<uint64_t>(eval_threads);
    }
    return static_cast<uint64_t>(eval_threads) * static_cast<uint64_t>(kMleItemsPerThread);
}
