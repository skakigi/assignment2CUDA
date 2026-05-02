source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_LIB=$(python - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).parent / "lib")
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
