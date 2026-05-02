# CUDA SumCheck current implementation

This package is a reconstructed current CUDA implementation from the project-only chat context. I did **not** find an uploaded `sumcheck_native.cu`, `student.cu`, or assignment harness source file in the project file library, so this zip provides a clean drop-in implementation that matches the direction of the previous work:

- native CUDA instead of JAX, to avoid JIT issues in the benchmark path;
- challenge-driven multi-round SumCheck;
- product-of-MLE tables with per-round univariate evaluations;
- fused round evaluation + MLE update kernels;
- a pure Python reference fallback for correctness/debugging;
- a PyTorch CUDA extension path for H100/Colab style runs.

The implemented polynomial is:

```text
g(x) = prod_j table_j(x)
```

where every `table_j` is a multilinear-extension evaluation table of length `2^v`. For `k` input tables, every round emits `k + 1` evaluations of the round polynomial at points `0, 1, ..., k`, then binds the current variable to the supplied challenge.

The MLE update convention is adjacent-pair / least-significant-variable first:

```text
t'[i] = t[2*i] + r * (t[2*i + 1] - t[2*i]) mod p
```

This is the same ordering that came up in the prior discussion and matches the common CUDA-friendly table-halving layout.

## Contents

```text
student.py                         Drop-in Python API: sumcheck(eval_tables, challenges, modulus=None)
setup.py                           Build native CUDA extension
pyproject.toml                     Minimal build metadata
src/sumcheck_native.cu             CUDA/C++ PyTorch extension
src/sumcheck_cpu_reference.py      Small, readable reference implementation
scripts/build_extension.py         Builds the extension in-place
scripts/install_in_harness.py      Copies the implementation into an existing assignment directory
scripts/run_smoke.py               Quick sanity test
scripts/make_test_inputs.py        Generates small .npy inputs
notebooks/run_uploaded_directory.ipynb  Simple Colab-style notebook
 tests/test_correctness.py          Pytest correctness checks
```

## Quick start

From this directory:

```bash
python -m pip install -e .
python scripts/build_extension.py
python scripts/run_smoke.py
pytest -q
```

On Colab/H100, use:

```bash
cd /content/cuda_sumcheck_current
python scripts/build_extension.py
python scripts/run_smoke.py --cuda
```

## Drop into an assignment harness

If your harness expects `student.py` at the project root:

```bash
python scripts/install_in_harness.py /content/sumcheck_repo/assignment2
cd /content/sumcheck_repo/assignment2
python setup.py build_ext --inplace
python -m pytest -q
```

If your harness has a native source directory, `install_in_harness.py` will also copy `src/sumcheck_native.cu` into it. If not, it leaves the CUDA file under `src/` and uses `setup.py` from the project root.

## Python API

```python
import student
out = student.sumcheck(eval_tables, challenges)
```

Input forms:

- NumPy arrays, Python lists, or Torch tensors.
- Shape of `eval_tables`: `(num_tables, table_len)` where `table_len` is a power of two.
- Shape of `challenges`: `(log2(table_len),)`.
- Field values are interpreted modulo `modulus`.

Default modulus:

```text
18446744069414584321  # Goldilocks: 2^64 - 2^32 + 1
```

For compatibility with signed int64-only harnesses, you can pass a smaller modulus:

```python
student.sumcheck(tables, challenges, modulus=2305843009213693951)  # 2^61 - 1
```

## Notes and limitations

- The native extension currently supports `torch.uint64` CUDA tensors.
- Modular multiplication uses CUDA device `unsigned __int128`, compiled with `--device-int128`. This is intentionally simple and reliable for correctness; it is not the final fastest Montgomery/Barrett kernel.
- The CUDA code supports up to 32 input MLE tables by default (`MAX_DEGREE = 32`). Raise that constant if needed.
- This is not a full HyperPlonk prover. It is the SumCheck kernel path needed for the assignment-style `student.sumcheck(eval_tables, challenges)` benchmark.
