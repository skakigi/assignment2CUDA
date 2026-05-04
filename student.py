"""Drop-in student API for the CUDA SumCheck implementation.

Primary entry point:
    sumcheck(eval_tables, challenges, modulus=None)

The native path uses a PyTorch CUDA extension when the input is a CUDA tensor.
The fallback path is the readable Python reference, which is slow but useful for
small correctness tests and CPU-only environments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.sumcheck_cpu_reference import FIELD_MODULUS, sumcheck_reference

_NATIVE = None
_NATIVE_LOAD_ERROR: Optional[BaseException] = None


def _try_import_native():
    global _NATIVE, _NATIVE_LOAD_ERROR
    if _NATIVE is not None:
        return _NATIVE
    if os.environ.get("SUMCHECK_USE_NATIVE", "1") == "0":
        return None
    try:
        import sumcheck_cuda_ext  # type: ignore

        _NATIVE = sumcheck_cuda_ext
        return _NATIVE
    except BaseException as exc:  # import can fail before build
        _NATIVE_LOAD_ERROR = exc
        return None


def _try_jit_native():
    """JIT-build the CUDA extension if it was not prebuilt.

    This is convenient in notebooks. For benchmark runs, prefer
    `python setup.py build_ext --inplace` first so import time is stable.
    """
    global _NATIVE, _NATIVE_LOAD_ERROR
    if _NATIVE is not None:
        return _NATIVE
    if os.environ.get("SUMCHECK_JIT_BUILD", "1") == "0":
        return None
    try:
        import torch
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        src = root / "src" / "sumcheck_native.cu"
        _NATIVE = load(
            name="sumcheck_cuda_ext",
            sources=[str(src)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math",  "-lineinfo", "-std=c++17"],
            verbose=bool(int(os.environ.get("SUMCHECK_VERBOSE_BUILD", "0"))),
        )
        return _NATIVE
    except BaseException as exc:
        _NATIVE_LOAD_ERROR = exc
        return None


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _torch_to_uint64_cuda(x, device):
    import torch

    if x.dtype != torch.uint64:
        x = x.to(dtype=torch.uint64)
    if x.device != device:
        x = x.to(device=device)
    return x.contiguous()


def _native_sumcheck(eval_tables, challenges, modulus: int):
    import torch

    if not isinstance(eval_tables, torch.Tensor):
        return None
    if not eval_tables.is_cuda:
        return None
    native = _try_import_native() or _try_jit_native()
    if native is None:
        return None
    tables = _torch_to_uint64_cuda(eval_tables, eval_tables.device)
    rs = challenges if isinstance(challenges, torch.Tensor) else torch.as_tensor(challenges)
    rs = _torch_to_uint64_cuda(rs, eval_tables.device)
    # Some benchmark-focused builds export only the explicit u32/u64/u128
    # entry points, not the legacy simple `sumcheck_cuda` API.
    # Keep this consistent with the other native paths: use native only
    # when the expected symbol exists, otherwise fall back to reference.
    if not hasattr(native, "sumcheck_cuda"):
        return None
    return native.sumcheck_cuda(tables, rs, int(modulus))


def _to_numpy_for_reference(x: Any) -> np.ndarray:
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy()
    # JAX and CuPy arrays usually implement __array__ after device-to-host copy.
    return np.asarray(x)


def _wrap_like_input(out, like):
    arr = np.asarray(out, dtype=object)
    if _is_torch_tensor(like):
        import torch

        # Python object arrays cannot be directly converted to torch. Use nested list.
        # Values above signed int64 require uint64.
        dtype = torch.uint64 if getattr(like, "dtype", None) == torch.uint64 else torch.long
        return torch.tensor(out, dtype=dtype, device=like.device)
    try:
        import jax.numpy as jnp  # type: ignore

        mod_name = type(like).__module__
        if mod_name.startswith("jax"):
            return jnp.asarray(np.asarray(out, dtype=np.uint64))
    except Exception:
        pass
    return np.asarray(out, dtype=np.uint64)


def sumcheck(eval_tables, challenges, modulus: Optional[int] = None):
    """Compute SumCheck transcript for a product of MLE tables.

    Args:
        eval_tables: `(num_tables, 2**rounds)` list/NumPy/Torch/JAX-like array.
        challenges: `(rounds,)` verifier challenges.
        modulus: field modulus. Defaults to Goldilocks.

    Returns:
        `(rounds, num_tables + 1)` transcript. For a Torch CUDA input and a
        successfully built extension, this is a CUDA `torch.uint64` tensor.
        Otherwise a CPU fallback result is returned in an input-like container.
    """
    p = int(FIELD_MODULUS if modulus is None else modulus)

    native_out = _native_sumcheck(eval_tables, challenges, p) if _is_torch_tensor(eval_tables) else None
    if native_out is not None:
        return native_out

    tables_np = _to_numpy_for_reference(eval_tables)
    challenges_np = _to_numpy_for_reference(challenges)
    if tables_np.ndim != 2:
        raise ValueError(f"eval_tables must be rank-2, got shape {tables_np.shape}")
    if challenges_np.ndim != 1:
        challenges_np = challenges_np.reshape(-1)
    out = sumcheck_reference(tables_np.tolist(), challenges_np.tolist(), p)
    return _wrap_like_input(out, eval_tables)


def native_status() -> str:
    native = _try_import_native()
    if native is not None:
        return "native extension import OK"
    if _NATIVE_LOAD_ERROR is None:
        return "native extension not attempted"
    return f"native extension unavailable: {_NATIVE_LOAD_ERROR}"


_EXPR_TO_ID = {
    "a": 0,
    "a*b": 1,
    "a*b + c": 2,
    "a*b+c": 2,
    "a*b*c": 3,
}

def sumcheck_expr(eval_tables, challenges, modulus: Optional[int] = None, expr: str = "a*b"):
    """Fused native SumCheck for the benchmark expression family.

    Supported:
        a
        a*b
        a*b + c
        a*b*c

    For Torch CUDA inputs, this calls one native CUDA extension function.
    """
    p = int(FIELD_MODULUS if modulus is None else modulus)
    if expr not in _EXPR_TO_ID:
        raise ValueError(f"unsupported expr {expr!r}; expected one of {sorted(_EXPR_TO_ID)}")

    if _is_torch_tensor(eval_tables):
        import torch
        if eval_tables.is_cuda:
            native = _try_import_native() or _try_jit_native()
            if native is not None and hasattr(native, "sumcheck_expr_cuda"):
                tables = _torch_to_uint64_cuda(eval_tables, eval_tables.device)
                rs = challenges if isinstance(challenges, torch.Tensor) else torch.as_tensor(challenges)
                rs = _torch_to_uint64_cuda(rs, eval_tables.device)
                return native.sumcheck_expr_cuda(tables, rs, int(p), int(_EXPR_TO_ID[expr]))

    # CPU fallback for small tests.
    tables_np = _to_numpy_for_reference(eval_tables)
    challenges_np = _to_numpy_for_reference(challenges)
    rows = tables_np.tolist()
    rs = challenges_np.reshape(-1).tolist()

    if expr == "a":
        out = sumcheck_reference([rows[0]], rs, p)
    elif expr == "a*b":
        out = sumcheck_reference([rows[0], rows[1]], rs, p)
    elif expr == "a*b*c":
        out = sumcheck_reference([rows[0], rows[1], rows[2]], rs, p)
    elif expr in ("a*b + c", "a*b+c"):
        ab = sumcheck_reference([rows[0], rows[1]], rs, p)
        c = sumcheck_reference([rows[2]], rs, p)
        out = []
        for ab_row, c_row in zip(ab, c):
            c0, c1 = int(c_row[0]), int(c_row[1])
            c2 = (2 * c1 - c0) % p
            out.append([
                (int(ab_row[0]) + c0) % p,
                (int(ab_row[1]) + c1) % p,
                (int(ab_row[2]) + c2) % p,
            ])
    else:
        raise ValueError(expr)

    return _wrap_like_input(out, eval_tables)

def _expr_to_terms_uploaded_base(expr: str):
    expr = expr.replace(' ', '')
    if expr == 'a':
        return [0, 1], [0]
    if expr == 'a*b':
        return [0, 2], [0, 1]
    if expr == 'a*b+c':
        return [0, 2, 3], [0, 1, 2]
    if expr == 'a*b*c':
        return [0, 3], [0, 1, 2]
    raise ValueError(f'unsupported expr {expr!r}')


def sumcheck_terms_u32(eval_tables, challenges, modulus, term_offsets, term_vars):
    import torch

    # Maintained u32 CUDA path: full Montgomery over the assignment 32-bit prime.
    # The old arbitrary-modulus uploaded-base u32 implementation has been retired.
    if int(modulus) != 4294967291:
        raise ValueError(
            "u32 CUDA SumCheck now requires modulus 4294967291 "
            "(full-Montgomery assignment-prime path)"
        )

    native = _try_import_native() or _try_jit_native()
    if native is None or not hasattr(native, 'sumcheck_terms_full_mont_u32_cuda'):
        raise RuntimeError('native full-Montgomery u32 CUDA extension is unavailable')

    if not _is_torch_tensor(eval_tables):
        tables = torch.as_tensor(eval_tables, device='cuda')
    else:
        tables = eval_tables
    if not tables.is_cuda:
        tables = tables.cuda()
    if tables.dtype != torch.uint32:
        tables = tables.to(torch.uint32)
    tables = tables.contiguous()

    if not _is_torch_tensor(challenges):
        rs = torch.as_tensor(challenges, device=tables.device)
    else:
        rs = challenges.to(tables.device)
    if rs.dtype != torch.uint32:
        rs = rs.to(torch.uint32)
    rs = rs.contiguous()

    offs = torch.as_tensor(term_offsets, dtype=torch.int32)
    vars_ = torch.as_tensor(term_vars, dtype=torch.int32)
    claim0, round_evals = native.sumcheck_terms_full_mont_u32_cuda(
        tables, rs, offs, vars_, int(modulus)
    )
    return claim0, round_evals


def sumcheck_expr(eval_tables, challenges, modulus=None, expr: str = 'a*b'):
    # Preserve the benchmark-facing API: return round_evals only.
    p = int(FIELD_MODULUS if modulus is None else modulus)
    offs, vars_ = _expr_to_terms_uploaded_base(expr)
    _claim0, round_evals = sumcheck_terms_u32(eval_tables, challenges, p, offs, vars_)
    return round_evals   
