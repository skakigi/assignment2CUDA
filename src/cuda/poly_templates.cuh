#pragma once

// Fixed-template polynomial IDs and metadata.
// Included inside the same namespace as the original definitions.

enum PolyTemplateId : int {
    kPolyVanillaGate = 0,
    kPolyVanillaZero = 1,
    kPolyVanillaPerm = 2,
    kPolyOpencheck6 = 3,

    kPolyDegreeSweepDeg3 = 5,
    kPolyDegreeSweepDeg5 = 6,
    kPolyDegreeSweepDeg7 = 7,

    kPolyBaselineLinear = 8,
    kPolyBaselineMul = 9,
    kPolyBaselineMulAdd = 10,
    kPolyBaselineCubicProduct = 11,

    kPolyAdvancedA2B2C = 12,
    kPolyAdvancedAbcPlusDe = 13,
    kPolyAdvancedAbcgPlusDeg = 14,
};

struct PolyTemplateMeta {
    int degree;
    int rows;
    const char* name;
};

inline PolyTemplateMeta meta_for_poly(int poly_id) {
    switch (poly_id) {
        case kPolyVanillaGate:
            return {3, 8, "vanilla_gate"};
        case kPolyVanillaZero:
            return {4, 9, "vanilla_zero"};
        case kPolyVanillaPerm:
            return {5, 11, "vanilla_perm"};
        case kPolyOpencheck6:
            return {1, 6, "opencheck_6"};

        case kPolyDegreeSweepDeg3:
            return {3, 6, "degree_sweep_deg3"};
        case kPolyDegreeSweepDeg5:
            return {5, 6, "degree_sweep_deg5"};
        case kPolyDegreeSweepDeg7:
            return {7, 6, "degree_sweep_deg7"};

        case kPolyBaselineLinear:
            return {1, 1, "baseline_linear"};
        case kPolyBaselineMul:
            return {2, 2, "baseline_mul"};
        case kPolyBaselineMulAdd:
            return {2, 3, "baseline_mul_add"};
        case kPolyBaselineCubicProduct:
            return {3, 3, "baseline_cubic_product"};

        case kPolyAdvancedA2B2C:
            return {5, 3, "advanced_a2b2c"};
        case kPolyAdvancedAbcPlusDe:
            return {3, 5, "advanced_abc_plus_de"};
        case kPolyAdvancedAbcgPlusDeg:
            return {4, 6, "advanced_abcg_plus_deg"};

        default:
            TORCH_CHECK(false, "unsupported fixed-template poly_id: ", poly_id);
    }

    return {-1, -1, "unsupported"};
}

int degree_for_poly(int poly_id) {
    return meta_for_poly(poly_id).degree;
}

int rows_for_poly(int poly_id) {
    return meta_for_poly(poly_id).rows;
}
