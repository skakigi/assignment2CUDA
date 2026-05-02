from pathlib import Path
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except Exception as exc:  # pragma: no cover
    CUDAExtension = None
    BuildExtension = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parent

if CUDAExtension is None:
    raise RuntimeError(
        "PyTorch with CUDA extension support is required to build sumcheck_cuda_ext. "
        f"Original import error: {_IMPORT_ERROR}"
    )

setup(
    name="sumcheck_cuda_ext",
    version="0.3.0",
    ext_modules=[
        CUDAExtension(
            name="sumcheck_cuda_ext",
            sources=[str(ROOT / "src" / "sumcheck_native.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    
                    "-lineinfo",
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
