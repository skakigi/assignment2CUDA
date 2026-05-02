#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct ScRunTimings {
    float h2d_ms;
    float kernel_ms;
    float d2h_ms;
    float total_ms;
};

int sc_sumcheck_u32(
    const uint32_t* const* host_inputs,
    int32_t n_inputs,
    const int32_t* term_offsets,
    const int32_t* term_vars,
    int32_t n_terms,
    uint64_t n,
    uint32_t q,
    const uint32_t* host_challenges,
    int32_t n_prover_challenges,
    int32_t num_rounds,
    uint32_t* host_claim0_out,
    uint32_t* host_round_evals_out,
    ScRunTimings* timings_out);

const char* sc_last_error_message(void);

#ifdef __cplusplus
}
#endif