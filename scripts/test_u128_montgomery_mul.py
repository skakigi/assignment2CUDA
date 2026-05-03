import sys
import numpy as np
import torch

sys.path.insert(0, "src")
import sumcheck_cuda_ext

Q = (1 << 128) - 159
MASK64 = (1 << 64) - 1

rng = np.random.default_rng(0)
n = 20_000

def split_u128(x):
    return np.uint64(x & MASK64), np.uint64((x >> 64) & MASK64)

a_lo = np.empty(n, dtype=np.uint64)
a_hi = np.empty(n, dtype=np.uint64)
b_lo = np.empty(n, dtype=np.uint64)
b_hi = np.empty(n, dtype=np.uint64)

a_vals = []
b_vals = []

lo_raw = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
hi_raw = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
lo2_raw = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
hi2_raw = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)

for i in range(n):
    a = ((int(hi_raw[i]) << 64) | int(lo_raw[i])) % Q
    b = ((int(hi2_raw[i]) << 64) | int(lo2_raw[i])) % Q

    a_vals.append(a)
    b_vals.append(b)

    a_lo[i], a_hi[i] = split_u128(a)
    b_lo[i], b_hi[i] = split_u128(b)

a_lo_t = torch.from_numpy(a_lo).cuda()
a_hi_t = torch.from_numpy(a_hi).cuda()
b_lo_t = torch.from_numpy(b_lo).cuda()
b_hi_t = torch.from_numpy(b_hi).cuda()

out_lo_t, out_hi_t = sumcheck_cuda_ext.montgomery_u128_mul_test_cuda(
    a_lo_t,
    a_hi_t,
    b_lo_t,
    b_hi_t,
)

out_lo = out_lo_t.cpu().numpy()
out_hi = out_hi_t.cpu().numpy()

for i in range(n):
    got = (int(out_hi[i]) << 64) | int(out_lo[i])
    expected = (a_vals[i] * b_vals[i]) % Q

    if got != expected:
        raise AssertionError(
            f"mismatch at {i}: "
            f"a={a_vals[i]} b={b_vals[i]} "
            f"got={got} expected={expected}"
        )

print(f"u128 Montgomery multiplication smoke OK for {n} random products")
