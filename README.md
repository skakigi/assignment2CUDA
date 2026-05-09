# sumcheckCUDA

CUDA implementation and benchmarking framework for the prover-side SumCheck kernel over multilinear extension (MLE) tables.

This project focuses on accelerating the core SumCheck computation used in modern zero-knowledge proof systems such as HyperPlonk-style protocols. The implementation is not a full ZKP prover; it is a CUDA/PyTorch native extension for evaluating SumCheck rounds, folding MLE tables, and benchmarking different finite-field arithmetic backends.

---

## What this project implements

The project implements SumCheck over products and sums of multilinear-extension tables.

Given MLE tables such as:

```text
A, B, C, ...
```

the implementation can evaluate composed polynomials such as:

```text
A * B * C
```

or HyperPlonk-style sum-of-products expressions such as:

```text
qL*w1 + qR*w2 + qM*w1*w2 - qO*w3 + qC
```

The core SumCheck loop repeatedly:

1. pairs MLE table entries,
2. evaluates each pair as a line in the current variable,
3. computes product/sum polynomial values,
4. accumulates round polynomial evaluations,
5. reduces partial sums across CUDA threads,
6. folds the MLE tables using the round challenge.

---

## Background: MLE tables and pairs

An MLE table stores the evaluations of a multilinear polynomial on all Boolean inputs.

For `v` variables, an MLE table has:

```text
2^v entries
```

For example, with `v = 3`:

```text
index    Boolean point
0        000
1        001
2        010
3        011
4        100
5        101
6        110
7        111
```

During one SumCheck round, the implementation processes one variable. The table is split into two halves:

```text
[t0 t1 t2 t3 | t4 t5 t6 t7]
              half = 4
```

The CUDA implementation pairs entries as:

```text
(t0, t4)
(t1, t5)
(t2, t6)
(t3, t7)
```

For pair index `i`, this is:

```text
left  = t[i]
right = t[i + half]
```

These two values correspond to the current variable being `0` and `1`, while the remaining variables stay fixed.

The line through the pair is:

```text
t_x = t[i] + x * (t[i + half] - t[i])
```

This formula is used in two places:

1. to evaluate the SumCheck round polynomial at `x = 0, 1, ..., degree`,
2. to fold the table with the verifier challenge `r`.

The fold step is:

```text
new_t[i] = t[i] + r * (t[i + half] - t[i])
```

After each round, the table length is halved.

---

## CUDA implementation overview

The CUDA kernels are organized around the SumCheck round structure.

For each round:

```text
for each pair index i:
    for each evaluation point x = 0..degree:
        evaluate each MLE table at x using the pair
        multiply values according to the polynomial
        accumulate into g_j(x)
```

### Per-thread register accumulation

Each CUDA thread handles one or more pair indices `i`.

For a degree-`d` polynomial, the round polynomial needs:

```text
g_j(0), g_j(1), ..., g_j(d)
```

So each thread keeps local accumulators:

```text
local[0] = this thread's partial sum for g_j(0)
local[1] = this thread's partial sum for g_j(1)
...
local[d] = this thread's partial sum for g_j(d)
```

After each thread computes its local contribution, the implementation reduces across:

1. threads in a warp,
2. warps in a block,
3. blocks in the grid.

This avoids global atomics in the hot loop.

### MLE update / folding

After the round polynomial is computed, the challenge is used to fold each MLE table:

```text
new_t[i] = t[i] + r * (t[i + half] - t[i])
```

This produces the MLE tables for the next round.

---

## Arithmetic backends

The SumCheck algorithm is the same across all field-width variants. What changes is the field arithmetic backend.

| Field width | Storage per field element | Multiplication strategy | Implementation impact |
|---|---|---|---|
| 32-bit | one `uint32_t` | `uint32 × uint32 -> uint64` intermediate | simplest arithmetic path |
| 64-bit | one `uint64_t` | `uint64 × uint64 -> 128-bit` product using hi/lo halves | uses explicit high-half multiplication such as `__umul64hi` |
| 128-bit | two 64-bit limbs: `lo + hi` | `128 × 128 -> 256-bit` intermediate using multiple 64×64 partial products | requires multi-limb arithmetic and multi-limb Montgomery reduction |

The full-Montgomery paths convert inputs into Montgomery form, run SumCheck rounds in Montgomery form, and convert outputs back afterward.

---

## Hashed challenge mode

The benchmark can optionally include hashed challenge generation.

In the hashed version, challenges are generated Fiat-Shamir style:

```text
round evaluations -> SHA3 transcript -> next challenge
```

That means the next challenge is derived from the transcript and the round polynomial evaluations, not simply from the previous challenge alone.

---

## Repository structure

Typical important files:

```text
src/
  sumcheck_native.cu
  cuda/
    sumcheck_u32_full_mont.cuh
    sumcheck_u64_full_mont.cuh
    sumcheck_u128_full_mont.cuh
    poly_templates.cuh

scripts/
  build_extension.py
  global_benchmark_full_mont.py
  hashed_challenge_sumcheck_u64.py

tests/
  ...
```

The exact structure may change as the project evolves, but the main implementation lives under `src/` and the benchmark entrypoints live under `scripts/`.

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel ninja pytest pandas matplotlib rich
```

Load the CUDA environment:

```bash
source env_cuda.sh
```

Build the extension:

```bash
python scripts/build_extension.py
```

Or build directly:

```bash
python setup.py build_ext --inplace
```

---

## Running tests

Run the test suite with:

```bash
pytest
```

For quick CUDA/import validation:

```bash
python - <<'PY'
import sumcheck_cuda_ext
print("Loaded:", sumcheck_cuda_ext.__file__)
print([name for name in dir(sumcheck_cuda_ext) if "sumcheck" in name or "mont" in name])
PY
```

---

## Benchmarking

The main benchmark command is:

```bash
python scripts/global_benchmark_full_mont.py \
  --bits 32,64,128 \
  --num-vars 4,16,20 \
  --warmup 3 \
  --runs 10 \
  --check \
  --include-hashed
```

Options:

```text
--bits
    Which arithmetic backends to benchmark.
    Example: 32,64,128

--num-vars
    Number of SumCheck variables.
    A value v means each MLE table initially has 2^v entries.

--warmup
    Number of warmup runs before timing.

--runs
    Number of measured runs.

--check
    Compare against correctness checks where supported.

--include-hashed
    Include benchmarks with transcript-based hashed challenge generation.
```

For the assignment benchmark configuration, use:

```bash
python scripts/global_benchmark_full_mont.py \
  --bits 32,64,128 \
  --num-vars 4,16,20 \
  --warmup 3 \
  --runs 10 \
  --check \
  --include-hashed
```

---

## Expected outputs

The benchmark reports timing results for different:

```text
field widths
problem sizes
polynomial templates
hashed vs non-hashed challenge modes
```

Depending on the script configuration, it may also generate CSV files, summary tables, or plots.

When presenting results, be clear that the measurements are for the SumCheck kernel implementation, not a full end-to-end HyperPlonk prover.

---

## Key CUDA optimizations

The implementation uses several CUDA-specific optimizations:

### 1. Split-half MLE pairing

The CUDA layout pairs:

```text
t[i] with t[i + half]
```

This gives coalesced reads from the left half and right half of the MLE table.

### 2. Per-thread register accumulation

Each thread accumulates its partial contribution to:

```text
g_j(0), g_j(1), ..., g_j(d)
```

in local registers before participating in reduction.

### 3. Hierarchical reduction

The implementation avoids global atomics by reducing in stages:

```text
thread-local registers
    -> warp reduction
    -> block reduction
    -> global/block partial reduction
```

### 4. Parallel MLE folding

The MLE update is embarrassingly parallel:

```text
new_t[i] = t[i] + r * (t[i + half] - t[i])
```

Each thread updates one or more table entries.

### 5. Montgomery arithmetic

The full-Montgomery paths keep field elements in Montgomery form during the hot SumCheck loop to reduce modular multiplication overhead.

### 6. Specialized polynomial templates

Common polynomial structures can be handled with specialized code paths instead of fully generic term dispatch.

---

## Correctness testing

The CUDA SumCheck implementation has two correctness layers.

### 1. Pytest CPU-reference checks

Run:

```bash
python scripts/build_extension.py
pytest -q tests/test_correctness.py tests/test_cuda_full_mont_correctness.py

```

```text
This pytest checks:
baseline_linear
baseline_mul
baseline_mul_add
baseline_cubic_product
advanced_abc_plus_de
vanilla_gate
vanilla_zero
vanilla_perm
opencheck_6
degree_sweep_deg7

against a cpu implementation
```

## Limitations

This project is a SumCheck kernel implementation and benchmark framework.

It does not implement:

```text
full HyperPlonk proving
polynomial commitments
MSM commitments
complete Fiat-Shamir transcript for an entire proof system
verifier logic
end-to-end proof generation
```

The goal is to accelerate and study the SumCheck computation itself.

---

## Suggested presentation summary

A concise way to describe the project:

> This project implements a CUDA-accelerated SumCheck prover kernel over MLE tables. Each round pairs MLE entries, evaluates the induced multilinear lines at the required points, accumulates the round polynomial evaluations across CUDA threads, reduces the partial sums, and folds the MLE tables using the round challenge. The benchmark compares 32-bit, 64-bit, and 128-bit Montgomery arithmetic backends and optionally includes hashed transcript-based challenge generation.

---

## Acknowledgments

This project is inspired by SumCheck and MLE-based computations used in modern zero-knowledge proof systems, including HyperPlonk-style protocols and hardware-acceleration work such as zkSpeed, zkPHIRE, and NoCap.
