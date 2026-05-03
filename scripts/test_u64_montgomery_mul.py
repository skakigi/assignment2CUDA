import sys
import numpy as np
import torch

sys.path.insert(0, "src")
import sumcheck_cuda_ext

Q = (1 << 64) - (1 << 32) + 1

rng = np.random.default_rng(0)
n = 100_000

a_np = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64) % np.uint64(Q)
b_np = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64) % np.uint64(Q)

a = torch.from_numpy(a_np).cuda()
b = torch.from_numpy(b_np).cuda()

out = sumcheck_cuda_ext.montgomery_u64_mul_test_cuda(a, b).cpu().numpy()

for i in range(n):
    expected = (int(a_np[i]) * int(b_np[i])) % Q
    got = int(out[i])
    if got != expected:
        raise AssertionError(
            f"mismatch at {i}: a={int(a_np[i])} b={int(b_np[i])} "
            f"got={got} expected={expected}"
        )

print(f"u64 Montgomery multiplication smoke OK for {n} random products")
