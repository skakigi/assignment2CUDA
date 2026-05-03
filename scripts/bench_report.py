#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def clean_cell(x: str) -> str:
    return x.strip().replace(",", "")


def to_number(x: str):
    x = clean_cell(x)
    if x.endswith("x") and NUM_RE.match(x[:-1]):
        return float(x[:-1])
    if NUM_RE.match(x):
        if "." in x:
            return float(x)
        return int(x)
    return x


def infer_variant_from_path(path: Path) -> str:
    name = path.stem.lower()
    for variant in [
        "mle_tiled_warp",
        "tiledwarp",
        "mle_tiled_shared",
        "tiledshared",
        "reduce",
        "compiled",
        "baseline",
        "xconst",
        "tiled2",
        "tiled4",
        "auto",
    ]:
        if variant in name:
            return variant
    return "unknown"



def infer_challenge_mode_from_path(path: Path) -> str:
    name = path.stem.lower()
    if "unhashed" in name or "plain_challenge" in name or "fixed_challenge" in name:
        return "unhashed"
    if "hashed" in name or "sha3" in name or "transcript" in name:
        return "hashed"
    return "unknown"


def parse_logs(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []

    for path in paths:
        bits = None
        num_vars = None
        device = None
        variant = infer_variant_from_path(path)
        challenge_mode = infer_challenge_mode_from_path(path)
        current_header: list[str] | None = None

        for raw in path.read_text(errors="replace").splitlines():
            line = raw.rstrip()

            m = re.search(r"bits\s*=\s*(\d+)", line)
            if m:
                bits = int(m.group(1))

            m = re.search(r"num_vars\s*=\s*(\d+)", line)
            if m:
                num_vars = int(m.group(1))

            m = re.search(r"device:\s*(.+)$", line)
            if m:
                device = m.group(1).strip()

            m = re.search(r"challenge mode:\s*(.+)$", line)
            if m:
                text = m.group(1).strip()
                lowered = text.lower()
                if "sha3" in lowered or "transcript" in lowered or "hash" in lowered:
                    challenge_mode = "hashed"
                else:
                    challenge_mode = "unhashed"

            if "|" not in line:
                continue

            parts = [clean_cell(p) for p in line.split("|")]

            # Header row
            lowered = [p.lower() for p in parts]
            if ("template" in lowered or "poly" in lowered) and (
                "median_ms" in lowered or "spec_ms" in lowered
            ):
                current_header = [
                    p.lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("/", "_")
                    for p in parts
                ]
                continue

            # Separator row
            if set(line.replace("|", "").replace("+", "").strip()) <= {"-"}:
                continue

            if current_header is None:
                continue

            if len(parts) != len(current_header):
                continue

            rec = {k: to_number(v) for k, v in zip(current_header, parts)}
            rec["source_log"] = str(path)
            rec["variant"] = variant
            rec["challenge_mode"] = challenge_mode
            if bits is not None:
                rec["bits"] = bits
            if num_vars is not None:
                rec["num_vars"] = num_vars
            if device is not None:
                rec["device"] = device

            # Normalize template/poly naming.
            if "template" not in rec and "poly" in rec:
                rec["template"] = rec["poly"]

            rows.append(rec)

    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    fields: list[str] = []
    preferred = [
        "source_log",
        "device",
        "variant",
        "challenge_mode",
        "bits",
        "num_vars",
        "template",
        "poly",
        "function",
        "n",
        "rows",
        "terms",
        "deg",
        "generic_ms",
        "spec_ms",
        "median_ms",
        "p90_ms",
        "generic_p90",
        "spec_p90",
        "speedup",
        "generic_mpts_s",
        "spec_mpts_s",
        "shape",
        "first_chal",
        "transcript_prefix",
    ]

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    for k in preferred:
        if k in all_keys:
            fields.append(k)
            all_keys.remove(k)

    fields.extend(sorted(all_keys))

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_csv(csv_path: Path, out_dir: Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    if df.empty:
        raise SystemExit("CSV has no rows")

    if "template" not in df.columns:
        raise SystemExit("CSV lacks template/poly column")

    # Prefer spec/generic tables; fall back to median-only tables.
    if "spec_ms" in df.columns:
        ycols = [c for c in ["generic_ms", "spec_ms"] if c in df.columns]
    else:
        ycols = [c for c in ["median_ms", "p90_ms"] if c in df.columns]

    group_cols = [c for c in ["device", "bits", "num_vars", "variant", "challenge_mode"] if c in df.columns]

    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        title_parts = []
        for col, val in zip(group_cols, key):
            title_parts.append(f"{col}={val}")
        title = ", ".join(title_parts) if title_parts else "benchmark"

        g = g.copy()
        g["template"] = g["template"].astype(str)

        # Runtime bar chart.
        ax = g.plot(
            x="template",
            y=ycols,
            kind="bar",
            figsize=(14, 6),
            rot=60,
            title=f"Runtime: {title}",
        )
        ax.set_ylabel("ms")
        ax.set_xlabel("")
        plt.tight_layout()

        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", title)
        plt.savefig(out_dir / f"runtime_{slug}.png", dpi=160)
        plt.close()

        # Speedup chart if available.
        if "speedup" in g.columns:
            ax = g.plot(
                x="template",
                y=["speedup"],
                kind="bar",
                figsize=(14, 5),
                rot=60,
                title=f"Speedup: {title}",
                legend=False,
            )
            ax.set_ylabel("x")
            ax.set_xlabel("")
            plt.tight_layout()
            plt.savefig(out_dir / f"speedup_{slug}.png", dpi=160)
            plt.close()

        # Throughput chart if available.
        tcols = [c for c in ["generic_mpts_s", "spec_mpts_s"] if c in g.columns]
        if tcols:
            ax = g.plot(
                x="template",
                y=tcols,
                kind="bar",
                figsize=(14, 6),
                rot=60,
                title=f"Throughput: {title}",
            )
            ax.set_ylabel("Mpts/s")
            ax.set_xlabel("")
            plt.tight_layout()
            plt.savefig(out_dir / f"throughput_{slug}.png", dpi=160)
            plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse")
    p_parse.add_argument("logs", nargs="+", type=Path)
    p_parse.add_argument("--out", type=Path, default=Path("reports/benchmarks.csv"))

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--csv", type=Path, default=Path("reports/benchmarks.csv"))
    p_plot.add_argument("--out-dir", type=Path, default=Path("reports/plots"))

    args = ap.parse_args()

    if args.cmd == "parse":
        rows = parse_logs(args.logs)
        write_csv(rows, args.out)
        print(f"wrote {len(rows)} rows to {args.out}")
    elif args.cmd == "plot":
        plot_csv(args.csv, args.out_dir)
        print(f"wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
