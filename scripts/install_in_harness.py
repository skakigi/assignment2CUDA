#!/usr/bin/env python3
"""Copy the implementation into an assignment harness directory.

This avoids assuming the scaffold's exact layout. It always writes:
    <harness>/student.py
    <harness>/setup.py
    <harness>/pyproject.toml
    <harness>/src/sumcheck_native.cu
    <harness>/src/sumcheck_cpu_reference.py

If the harness already contains a file named sumcheck_native.cu elsewhere, the
script also overwrites that file so native-source based scaffolds pick it up.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src.relative_to(ROOT)} -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("harness_root", type=Path)
    args = ap.parse_args()
    dst = args.harness_root.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    copy(ROOT / "student.py", dst / "student.py")
    copy(ROOT / "setup.py", dst / "setup.py")
    copy(ROOT / "pyproject.toml", dst / "pyproject.toml")
    copy(ROOT / "src" / "sumcheck_native.cu", dst / "src" / "sumcheck_native.cu")
    copy(ROOT / "src" / "sumcheck_cpu_reference.py", dst / "src" / "sumcheck_cpu_reference.py")

    existing_native = list(dst.rglob("sumcheck_native.cu"))
    for path in existing_native:
        if path.resolve() != (dst / "src" / "sumcheck_native.cu").resolve():
            copy(ROOT / "src" / "sumcheck_native.cu", path)

    print("\nNext steps:")
    print(f"  cd {dst}")
    print("  python setup.py build_ext --inplace")
    print("  python - <<'PY'\nimport student; print(student.native_status())\nPY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
