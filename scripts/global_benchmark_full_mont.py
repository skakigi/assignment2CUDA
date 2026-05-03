#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_SCRIPT = ROOT / "scripts" / "paper_poly_compare_full_mont.py"
HASHED_SCRIPT = ROOT / "scripts" / "hashed_challenge_sumcheck_u64.py"

DEFAULT_POLYS = [
    "baseline_linear",
    "baseline_mul",
    "baseline_mul_add",
    "baseline_cubic_product",
    "advanced_a2b2c",
    "advanced_abc_plus_de",
    "advanced_abcg_plus_deg",
    "vanilla_gate",
    "vanilla_zero",
    "vanilla_perm",
    "opencheck_6",
    "degree_sweep_deg3",
    "degree_sweep_deg5",
    "degree_sweep_deg7",
]

TEMPLATE_META = {
    "baseline_linear": ("a", 1, 1),
    "baseline_mul": ("a*b", 2, 1),
    "baseline_mul_add": ("a*b + c", 3, 2),
    "baseline_cubic_product": ("a*b*c", 3, 1),
    "advanced_a2b2c": ("a*a*b*b*c", 3, 1),
    "advanced_abc_plus_de": ("a*b*c + d*e", 5, 2),
    "advanced_abcg_plus_deg": ("a*b*c*g + d*e*g", 6, 2),
    "vanilla_gate": ("qL*w1 + qR*w2 + qM*w1*w2 - qO*w3 + qC", 8, 5),
    "vanilla_zero": ("(qL*w1 + qR*w2 + qM*w1*w2 - qO*w3 + qC)*fr", 9, 5),
    "vanilla_perm": ("(pi - p1*p2 + alpha_phi*D1*D2*D3 - alpha*N1*N2*N3)*fr", 11, 4),
    "opencheck_6": ("y1*k1 + y2*k2 + ... + y6*k6", 6, 6),
    "degree_sweep_deg3": ("q1*w1 + q2*w2 + qH*w1*w2 + qC", 6, 4),
    "degree_sweep_deg5": ("q1*w1 + q2*w2 + qH*w1^3*w2 + qC", 6, 4),
    "degree_sweep_deg7": ("q1*w1 + q2*w2 + qH*w1^5*w2 + qC", 6, 4),
}


@dataclass
class UnhashedRow:
    bits: str
    device: str
    backend: str
    num_vars: int
    template: str
    function: str
    N: int
    rows: int
    terms: int
    deg: int
    generic_ms: float
    spec_ms: float
    speedup: float
    generic_p90: float
    spec_p90: float
    generic_mpts_s: float
    spec_mpts_s: float
    shape: str


@dataclass
class HashedRow:
    bits: str
    device: str
    backend: str
    num_vars: int
    template: str
    function: str
    N: int
    rows: int
    terms: int
    deg: int
    median_ms: float
    p90_ms: float
    mpts_s: float
    first_chal: str
    transcript_prefix: str
    shape: str


def cell(x: str) -> str:
    return x.strip()


def header_name(x: str) -> str:
    return cell(x).lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def parse_val(x: str):
    x = cell(x)
    x_num = x.replace(",", "")
    if x_num.endswith("x"):
        try:
            return float(x_num[:-1])
        except ValueError:
            return x
    if re.fullmatch(r"-?\d+", x_num):
        return int(x_num)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", x_num):
        return float(x_num)
    return x


def fnum(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def fnum_any(rec: dict, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in rec:
            return fnum(rec.get(key), default)
    return default


def inum(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def inum_any(rec: dict, keys: list[str], default: int = 0) -> int:
    for key in keys:
        if key in rec:
            return inum(rec.get(key), default)
    return default


def mpts(N: int, ms: float) -> float:
    return (N / ms / 1000.0) if ms > 0 else 0.0


def speedup(generic_ms: float, spec_ms: float) -> float:
    return (generic_ms / spec_ms) if generic_ms > 0 and spec_ms > 0 else 0.0


def meta_for(template: str) -> tuple[str, int, int]:
    return TEMPLATE_META.get(template, ("", 0, 0))


def run_cmd(cmd: list[str]) -> str:
    print("running:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)
    return proc.stdout


def parse_pipe_table(text: str, required: set[str]) -> tuple[dict, list[dict]]:
    run_meta = {
        "device": "unknown",
        "backend": "unknown",
        "num_vars": -1,
    }
    rows: list[dict] = []
    headers: list[str] | None = None

    for raw in text.splitlines():
        line = raw.rstrip()

        m = re.search(r"device:\s*(.+)$", line)
        if m:
            run_meta["device"] = m.group(1).strip()

        m = re.search(r"backend:\s*(.+)$", line)
        if m:
            run_meta["backend"] = m.group(1).strip()

        m = re.search(r"num_vars\s*=\s*(\d+)", line)
        if m:
            run_meta["num_vars"] = int(m.group(1))

        if "|" not in line:
            continue

        parts = [cell(p) for p in line.split("|")]
        lowered = {p.lower() for p in parts}

        if required.issubset(lowered):
            headers = [header_name(p) for p in parts]
            continue

        if set(line.replace("|", "").replace("+", "").strip()) <= {"-"}:
            continue

        if headers is None or len(parts) != len(headers):
            continue

        rows.append({k: parse_val(v) for k, v in zip(headers, parts)})

    return run_meta, rows


def run_unhashed(args: argparse.Namespace, polys: str) -> tuple[str, list[UnhashedRow]]:
    cmd = [
        sys.executable,
        str(PAPER_SCRIPT),
        "--num-vars",
        str(args.num_vars),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--polys",
        polys,
    ]
    if args.check:
        cmd.append("--check")
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    raw = run_cmd(cmd)
    run_meta, parsed = parse_pipe_table(
        raw,
        {"template", "generic_ms", "spec_ms"},
    )
    if not parsed:
        run_meta, parsed = parse_pipe_table(
            raw,
            {"template", "full_generic_ms", "full_spec_ms"},
        )

    out: list[UnhashedRow] = []
    for rec in parsed:
        template = str(rec.get("template", ""))
        function, rows, terms = meta_for(template)

        N = inum_any(rec, ["n", "N"])
        deg = inum_any(rec, ["deg", "degree"])
        generic_ms = fnum_any(rec, ["generic_ms", "full_generic_ms"])
        spec_ms = fnum_any(rec, ["spec_ms", "full_spec_ms"])
        generic_p90 = fnum_any(
            rec,
            ["generic_p90", "generic_p90_ms", "full_generic_p90", "full_generic_p90_ms"],
        )
        spec_p90 = fnum_any(
            rec,
            ["spec_p90", "spec_p90_ms", "full_spec_p90", "full_spec_p90_ms"],
        )
        generic_p90 = fnum_any(rec, ["generic_p90", "generic_p90_ms", "full_generic_p90", "full_generic_p90_ms"])
        spec_p90 = fnum_any(rec, ["spec_p90", "spec_p90_ms", "full_spec_p90", "full_spec_p90_ms"])

        out.append(
            UnhashedRow(
                bits=args.bits,
                device=str(run_meta["device"]),
                backend=(
                    "full Montgomery paper compare"
                    if str(run_meta["backend"]) == "unknown"
                    else str(run_meta["backend"])
                ),
                num_vars=inum(run_meta["num_vars"]),
                template=template,
                function=function,
                N=N,
                rows=rows,
                terms=terms,
                deg=deg,
                generic_ms=generic_ms,
                spec_ms=spec_ms,
                speedup=speedup(generic_ms, spec_ms),
                generic_p90=generic_p90,
                spec_p90=spec_p90,
                generic_mpts_s=mpts(N, generic_ms),
                spec_mpts_s=mpts(N, spec_ms),
                shape=str(rec.get("shape", "")),
            )
        )

    if not out:
        raise SystemExit("No unhashed rows parsed.")

    return raw, out


def run_hashed(args: argparse.Namespace, polys: str) -> tuple[str, list[HashedRow]]:
    cmd = [
        sys.executable,
        str(HASHED_SCRIPT),
        "--num-vars",
        str(args.num_vars),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--polys",
        polys,
    ]
    if args.check:
        cmd.append("--check")
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    raw = run_cmd(cmd)
    run_meta, parsed = parse_pipe_table(
        raw,
        {"poly", "function", "n", "deg", "median_ms", "p90_ms"},
    )

    out: list[HashedRow] = []
    for rec in parsed:
        template = str(rec.get("poly", ""))
        function, rows, terms = meta_for(template)
        if not function:
            function = str(rec.get("function", ""))

        N = inum(rec.get("n"))
        deg = inum(rec.get("deg"))
        median_ms = fnum(rec.get("median_ms"))
        p90_ms = fnum(rec.get("p90_ms"))

        out.append(
            HashedRow(
                bits=args.bits,
                device=str(run_meta["device"]),
                backend=(
                    "full Montgomery paper compare"
                    if str(run_meta["backend"]) == "unknown"
                    else str(run_meta["backend"])
                ),
                num_vars=inum(run_meta["num_vars"]),
                template=template,
                function=function,
                N=N,
                rows=rows,
                terms=terms,
                deg=deg,
                median_ms=median_ms,
                p90_ms=p90_ms,
                mpts_s=mpts(N, median_ms),
                first_chal=str(rec.get("first_chal", "")),
                transcript_prefix=str(rec.get("transcript_prefix", "")),
                shape=f"({inum(run_meta['num_vars'])}, {deg + 1})",
            )
        )

    if not out:
        raise SystemExit("No hashed rows parsed.")

    return raw, out


def md_escape(x: object) -> str:
    return str(x).replace("|", "\\|")


def write_table(f, columns: list[str], rows: list[dict], labels: dict[str, str] | None = None) -> None:
    labels = labels or {}
    f.write("| " + " | ".join(labels.get(c, c) for c in columns) + " |\n")
    f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(md_escape(val))
        f.write("| " + " | ".join(vals) + " |\n")
    f.write("\n")



def fmt_fixed_row(values: list[object], widths: list[int]) -> str:
    out = []
    for value, width in zip(values, widths):
        text = str(value)
        if len(text) > width:
            text = text[: max(0, width - 1)] + "…"
        out.append(text.rjust(width))
    return " | ".join(out)


def fmt_fixed_sep(widths: list[int]) -> str:
    return "-+-".join("-" * w for w in widths)


def write_fixed_section(
    f,
    title: str,
    meta_lines: list[str],
    headers: list[str],
    rows: list[list[object]],
    widths: list[int],
) -> None:
    f.write(f"{title}\n")
    f.write("=" * len(title) + "\n\n")
    for line in meta_lines:
        f.write(f"{line}\n")
    f.write("\n")
    f.write(fmt_fixed_row(headers, widths) + "\n")
    f.write(fmt_fixed_sep(widths) + "\n")
    for row in rows:
        f.write(fmt_fixed_row(row, widths) + "\n")
    f.write("\n\n")


def write_text_report(path: Path, unhashed: list[UnhashedRow], hashed: list[HashedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write("Global Full-Montgomery SumCheck Benchmark\n")
        f.write("=========================================\n\n")

        unhashed_headers = [
            "template",
            "function",
            "N",
            "rows",
            "terms",
            "deg",
            "generic_ms",
            "spec_ms",
            "speedup",
            "generic_p90",
            "spec_p90",
            "generic_Mpts/s",
            "spec_Mpts/s",
            "shape",
        ]
        unhashed_widths = [24, 68, 10, 5, 5, 4, 12, 10, 8, 12, 10, 15, 13, 12]
        unhashed_rows = [
            [
                r.template,
                r.function,
                r.N,
                r.rows,
                r.terms,
                r.deg,
                f"{r.generic_ms:.3f}",
                f"{r.spec_ms:.3f}",
                f"{r.speedup:.2f}x",
                f"{r.generic_p90:.3f}",
                f"{r.spec_p90:.3f}",
                f"{r.generic_mpts_s:.2f}",
                f"{r.spec_mpts_s:.2f}",
                r.shape,
            ]
            for r in unhashed
        ]

        write_fixed_section(
            f,
            "Unhashed / Paper Compare",
            [
                f"bits: {unhashed[0].bits}",
                f"num_vars: {unhashed[0].num_vars}",
                f"device: {unhashed[0].device}",
                f"backend: {unhashed[0].backend}",
            ],
            unhashed_headers,
            unhashed_rows,
            unhashed_widths,
        )

        if hashed:
            hashed_headers = [
                "template",
                "function",
                "N",
                "rows",
                "terms",
                "deg",
                "median_ms",
                "p90_ms",
                "Mpts/s",
                "first_chal",
                "transcript_prefix",
                "shape",
            ]
            hashed_widths = [24, 68, 10, 5, 5, 4, 10, 10, 10, 20, 18, 12]
            hashed_rows = [
                [
                    r.template,
                    r.function,
                    r.N,
                    r.rows,
                    r.terms,
                    r.deg,
                    f"{r.median_ms:.3f}",
                    f"{r.p90_ms:.3f}",
                    f"{r.mpts_s:.2f}",
                    r.first_chal,
                    r.transcript_prefix,
                    r.shape,
                ]
                for r in hashed
            ]

            write_fixed_section(
                f,
                "Hashed / SHA3 Transcript",
                [
                    f"bits: {hashed[0].bits}",
                    f"num_vars: {hashed[0].num_vars}",
                    f"device: {hashed[0].device}",
                    f"backend: {hashed[0].backend}",
                ],
                hashed_headers,
                hashed_rows,
                hashed_widths,
            )



def fmt_fixed_row(values: list[object], widths: list[int]) -> str:
    out = []
    for value, width in zip(values, widths):
        text = str(value)
        if len(text) > width:
            text = text[: max(0, width - 1)] + "…"
        out.append(text.rjust(width))
    return " | ".join(out)


def fmt_fixed_sep(widths: list[int]) -> str:
    return "-+-".join("-" * w for w in widths)


def write_fixed_section(
    f,
    title: str,
    meta_lines: list[str],
    headers: list[str],
    rows: list[list[object]],
    widths: list[int],
) -> None:
    f.write(f"{title}\n")
    f.write("=" * len(title) + "\n\n")
    for line in meta_lines:
        f.write(f"{line}\n")
    f.write("\n")
    f.write(fmt_fixed_row(headers, widths) + "\n")
    f.write(fmt_fixed_sep(widths) + "\n")
    for row in rows:
        f.write(fmt_fixed_row(row, widths) + "\n")
    f.write("\n\n")


def write_text_report(path: Path, unhashed: list[UnhashedRow], hashed: list[HashedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write("Global Full-Montgomery SumCheck Benchmark\n")
        f.write("=========================================\n\n")

        unhashed_headers = [
            "template",
            "function",
            "N",
            "rows",
            "terms",
            "deg",
            "generic_ms",
            "spec_ms",
            "speedup",
            "generic_p90",
            "spec_p90",
            "generic_Mpts/s",
            "spec_Mpts/s",
            "shape",
        ]
        unhashed_widths = [24, 68, 10, 5, 5, 4, 12, 10, 8, 12, 10, 15, 13, 12]
        unhashed_rows = [
            [
                r.template,
                r.function,
                r.N,
                r.rows,
                r.terms,
                r.deg,
                f"{r.generic_ms:.3f}",
                f"{r.spec_ms:.3f}",
                f"{r.speedup:.2f}x",
                f"{r.generic_p90:.3f}",
                f"{r.spec_p90:.3f}",
                f"{r.generic_mpts_s:.2f}",
                f"{r.spec_mpts_s:.2f}",
                r.shape,
            ]
            for r in unhashed
        ]

        write_fixed_section(
            f,
            "Unhashed / Paper Compare",
            [
                f"bits: {unhashed[0].bits}",
                f"num_vars: {unhashed[0].num_vars}",
                f"device: {unhashed[0].device}",
                f"backend: {unhashed[0].backend}",
            ],
            unhashed_headers,
            unhashed_rows,
            unhashed_widths,
        )

        if hashed:
            hashed_headers = [
                "template",
                "function",
                "N",
                "rows",
                "terms",
                "deg",
                "median_ms",
                "p90_ms",
                "Mpts/s",
                "first_chal",
                "transcript_prefix",
                "shape",
            ]
            hashed_widths = [24, 68, 10, 5, 5, 4, 10, 10, 10, 20, 18, 12]
            hashed_rows = [
                [
                    r.template,
                    r.function,
                    r.N,
                    r.rows,
                    r.terms,
                    r.deg,
                    f"{r.median_ms:.3f}",
                    f"{r.p90_ms:.3f}",
                    f"{r.mpts_s:.2f}",
                    r.first_chal,
                    r.transcript_prefix,
                    r.shape,
                ]
                for r in hashed
            ]

            write_fixed_section(
                f,
                "Hashed / SHA3 Transcript",
                [
                    f"bits: {hashed[0].bits}",
                    f"num_vars: {hashed[0].num_vars}",
                    f"device: {hashed[0].device}",
                    f"backend: {hashed[0].backend}",
                ],
                hashed_headers,
                hashed_rows,
                hashed_widths,
            )


def write_md(path: Path, unhashed: list[UnhashedRow], hashed: list[HashedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Global Full-Montgomery SumCheck Benchmark\n\n")

        f.write("## Unhashed / Paper Compare\n\n")
        f.write(f"- bits: `{unhashed[0].bits}`\n")
        f.write(f"- num_vars: `{unhashed[0].num_vars}`\n")
        f.write(f"- device: `{unhashed[0].device}`\n")
        f.write(f"- backend: `{unhashed[0].backend}`\n\n")

        write_table(
            f,
            [
                "template",
                "function",
                "N",
                "rows",
                "terms",
                "deg",
                "generic_ms",
                "spec_ms",
                "speedup",
                "generic_p90",
                "spec_p90",
                "generic_mpts_s",
                "spec_mpts_s",
                "shape",
            ],
            [asdict(r) for r in unhashed],
            {
                "generic_mpts_s": "generic_Mpts/s",
                "spec_mpts_s": "spec_Mpts/s",
            },
        )

        if hashed:
            f.write("## Hashed / SHA3 Transcript\n\n")
            f.write(f"- bits: `{hashed[0].bits}`\n")
            f.write(f"- num_vars: `{hashed[0].num_vars}`\n")
            f.write(f"- device: `{hashed[0].device}`\n")
            f.write(f"- backend: `{hashed[0].backend}`\n\n")

            write_table(
                f,
                [
                    "template",
                    "function",
                    "N",
                    "rows",
                    "terms",
                    "deg",
                    "median_ms",
                    "p90_ms",
                    "mpts_s",
                    "first_chal",
                    "transcript_prefix",
                    "shape",
                ],
                [asdict(r) for r in hashed],
                {
                    "mpts_s": "Mpts/s",
                },
            )


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    dicts = [asdict(r) for r in rows]
    fields = list(dicts[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(dicts)


def write_summary(path: Path, unhashed: list[UnhashedRow], hashed: list[HashedRow]) -> None:
    speedups = [r.speedup for r in unhashed]
    best = max(unhashed, key=lambda r: r.speedup)
    worst = min(unhashed, key=lambda r: r.speedup)

    lines = [
        "# Global Benchmark Summary\n\n",
        "## Unhashed / Paper Compare\n\n",
        f"- rows: `{len(unhashed)}`\n",
        f"- mean speedup: `{statistics.mean(speedups):.3f}x`\n",
        f"- median speedup: `{statistics.median(speedups):.3f}x`\n",
        f"- best speedup: `{best.template}` `{best.speedup:.3f}x`\n",
        f"- weakest speedup: `{worst.template}` `{worst.speedup:.3f}x`\n\n",
    ]

    if hashed:
        medians = [r.median_ms for r in hashed]
        lines += [
            "## Hashed / SHA3 Transcript\n\n",
            f"- rows: `{len(hashed)}`\n",
            f"- mean median_ms: `{statistics.mean(medians):.3f}`\n",
            f"- median median_ms: `{statistics.median(medians):.3f}`\n",
        ]

    path.write_text("".join(lines))


def plot_runtime(path: Path, unhashed: list[UnhashedRow], hashed: list[HashedRow]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"plot skipped: {e}")
        return

    path.mkdir(parents=True, exist_ok=True)

    x = list(range(len(unhashed)))
    width = 0.38
    labels = [r.template for r in unhashed]

    plt.figure(figsize=(16, 6))
    plt.bar([i - width / 2 for i in x], [r.generic_ms for r in unhashed], width, label="generic")
    plt.bar([i + width / 2 for i in x], [r.spec_ms for r in unhashed], width, label="specialized")
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("ms")
    plt.title("Unhashed / Paper Compare Runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path / "unhashed_runtime.png", dpi=160)
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.bar(x, [r.speedup for r in unhashed])
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("speedup (x)")
    plt.title("Unhashed / Paper Compare Speedup")
    plt.tight_layout()
    plt.savefig(path / "unhashed_speedup.png", dpi=160)
    plt.close()

    if hashed:
        x = list(range(len(hashed)))
        labels = [r.template for r in hashed]
        plt.figure(figsize=(16, 6))
        plt.bar(x, [r.median_ms for r in hashed])
        plt.xticks(x, labels, rotation=60, ha="right")
        plt.ylabel("ms")
        plt.title("Hashed / SHA3 Transcript Median Runtime")
        plt.tight_layout()
        plt.savefig(path / "hashed_runtime.png", dpi=160)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="64", help="reporting label for now; cross-bit execution is added next")
    ap.add_argument("--num-vars", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--polys", default=",".join(DEFAULT_POLYS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--include-hashed", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    raw_dir = args.out_dir / "raw"
    plot_dir = args.out_dir / "plots"
    raw_dir.mkdir(parents=True, exist_ok=True)

    unhashed_raw, unhashed = run_unhashed(args, args.polys)
    (raw_dir / f"global_unhashed_bits{args.bits}_nv{args.num_vars}.txt").write_text(unhashed_raw)

    hashed: list[HashedRow] = []
    if args.include_hashed:
        hashed_raw, hashed = run_hashed(args, args.polys)
        (raw_dir / f"global_hashed_bits{args.bits}_nv{args.num_vars}.txt").write_text(hashed_raw)

    write_csv(args.out_dir / "global_benchmark_unhashed.csv", unhashed)
    if hashed:
        write_csv(args.out_dir / "global_benchmark_hashed.csv", hashed)
    write_md(args.out_dir / "global_benchmark.md", unhashed, hashed)
    write_text_report(args.out_dir / "global_benchmark.txt", unhashed, hashed)
    write_summary(args.out_dir / "global_benchmark_summary.md", unhashed, hashed)

    if not args.no_plot:
        plot_runtime(plot_dir, unhashed, hashed)

    print(f"wrote markdown: {args.out_dir / 'global_benchmark.md'}")
    print(f"wrote text:     {args.out_dir / 'global_benchmark.txt'}")
    print(f"wrote summary:  {args.out_dir / 'global_benchmark_summary.md'}")
    print(f"wrote csv:     {args.out_dir / 'global_benchmark_unhashed.csv'}")
    if hashed:
        print(f"wrote hashed:  {args.out_dir / 'global_benchmark_hashed.csv'}")
    if not args.no_plot:
        print(f"wrote plots:   {plot_dir}")


if __name__ == "__main__":
    main()
