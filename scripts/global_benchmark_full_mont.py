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
PAPER_BITS_SCRIPT = ROOT / "scripts" / "paper_poly_compare_bits.py"
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

SCALING_POLYS = [
    "baseline_mul",
    "vanilla_gate",
    "vanilla_perm",
]

PLOT_STANDARD_BITS = 64
PLOT_STANDARD_NUM_VARS = 16
PLOT_NUM_VARS_VALUES = [4, 16, 20]
PLOT_REPRESENTATIVE_POLYS = [
    "baseline_mul",
    "vanilla_gate",
    "vanilla_perm",
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
    bits: int
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
    bits: int
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
    first_chal: str
    transcript_prefix: str
    shape: str


def meta_for(template: str) -> tuple[str, int, int]:
    return TEMPLATE_META.get(template, ("", 0, 0))


def header_name(x: str) -> str:
    return x.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def parse_val(x: str):
    x = x.strip()
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


def inum(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def fnum_any(rec: dict, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in rec:
            return fnum(rec.get(key), default)
    return default


def inum_any(rec: dict, keys: list[str], default: int = 0) -> int:
    for key in keys:
        if key in rec:
            return inum(rec.get(key), default)
    return default


def mpts(N: int, ms: float) -> float:
    return (N / ms / 1000.0) if ms > 0 else 0.0


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


def parse_pipe_tables_by_bits(text: str, required: set[str]) -> tuple[dict, list[tuple[dict, dict]]]:
    run_meta = {"device": "unknown", "backend": "unknown", "num_vars": -1}
    current_bits: int | None = None
    headers: list[str] | None = None
    out: list[tuple[dict, dict]] = []

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

        m = re.search(r"bits\s*=\s*(\d+)", line)
        if m:
            current_bits = int(m.group(1))
            headers = None
            continue

        if "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        lowered = {p.lower() for p in parts}

        if required.issubset(lowered):
            headers = [header_name(p) for p in parts]
            continue

        if set(line.replace("|", "").replace("+", "").strip()) <= {"-"}:
            continue

        if headers is None or len(parts) != len(headers):
            continue

        rec = {k: parse_val(v) for k, v in zip(headers, parts)}
        row_meta = dict(run_meta)
        if current_bits is not None:
            row_meta["bits"] = current_bits
        out.append((row_meta, rec))

    return run_meta, out


def run_unhashed(args: argparse.Namespace, polys: str) -> tuple[str, list[UnhashedRow]]:
    cmd = [
        sys.executable,
        str(PAPER_BITS_SCRIPT),
        "--bits",
        args.bits,
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
    _meta, parsed = parse_pipe_tables_by_bits(raw, {"template", "generic_ms", "spec_ms"})

    rows: list[UnhashedRow] = []
    for meta, rec in parsed:
        template = str(rec.get("template", ""))
        function, default_rows, default_terms = meta_for(template)

        N = inum_any(rec, ["n", "N"])
        deg = inum_any(rec, ["deg", "degree"])
        generic_ms = fnum_any(rec, ["generic_ms", "full_generic_ms"])
        spec_ms = fnum_any(rec, ["spec_ms", "full_spec_ms"])
        generic_p90 = fnum_any(rec, ["generic_p90", "generic_p90_ms"])
        spec_p90 = fnum_any(rec, ["spec_p90", "spec_p90_ms"])

        rows.append(
            UnhashedRow(
                bits=inum(meta.get("bits")),
                device=str(meta.get("device", "unknown")),
                backend=str(meta.get("backend", "full Montgomery domain")),
                num_vars=inum(meta.get("num_vars")),
                template=template,
                function=str(rec.get("function", function)),
                N=N,
                rows=inum(rec.get("rows"), default_rows),
                terms=inum(rec.get("terms"), default_terms),
                deg=deg,
                generic_ms=generic_ms,
                spec_ms=spec_ms,
                speedup=fnum(rec.get("speedup"), generic_ms / spec_ms if spec_ms > 0 else 0.0),
                generic_p90=generic_p90,
                spec_p90=spec_p90,
                generic_mpts_s=fnum_any(rec, ["generic_mpts_s", "generic_mpts/s"], mpts(N, generic_ms)),
                spec_mpts_s=fnum_any(rec, ["spec_mpts_s", "spec_mpts/s"], mpts(N, spec_ms)),
                shape=str(rec.get("shape", f"({inum(meta.get('num_vars'))}, {deg + 1})")),
            )
        )

    if not rows:
        raise SystemExit("No unhashed rows parsed from paper_poly_compare_bits.py output.")

    return raw, rows


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
    run_meta, parsed = parse_pipe_tables_by_bits(
        "bits = 64\n" + raw,
        {"poly", "function", "n", "deg", "generic_ms", "spec_ms"},
    )

    rows: list[HashedRow] = []
    for meta, rec in parsed:
        template = str(rec.get("poly", ""))
        function, default_rows, default_terms = meta_for(template)

        N = inum(rec.get("n"))
        deg = inum(rec.get("deg"))
        generic_ms = fnum(rec.get("generic_ms"))
        spec_ms = fnum(rec.get("spec_ms"))
        generic_p90 = fnum(rec.get("generic_p90"))
        spec_p90 = fnum(rec.get("spec_p90"))
        num_vars = inum(meta.get("num_vars"), inum(run_meta.get("num_vars")))

        rows.append(
            HashedRow(
                bits=64,
                device=str(meta.get("device", run_meta.get("device", "unknown"))),
                backend=str(meta.get("backend", run_meta.get("backend", "u64 full Montgomery generic/spec round eval + fold"))),
                num_vars=num_vars,
                template=template,
                function=str(rec.get("function", function)),
                N=N,
                rows=default_rows,
                terms=default_terms,
                deg=deg,
                generic_ms=generic_ms,
                spec_ms=spec_ms,
                speedup=fnum(rec.get("speedup"), generic_ms / spec_ms if spec_ms > 0 else 0.0),
                generic_p90=generic_p90,
                spec_p90=spec_p90,
                generic_mpts_s=mpts(N, generic_ms),
                spec_mpts_s=mpts(N, spec_ms),
                first_chal=str(rec.get("first_chal", "")),
                transcript_prefix=str(rec.get("transcript_prefix", rec.get("sha3_prefix", ""))),
                shape=f"({num_vars}, {deg + 1})",
            )
        )

    if not rows:
        raise SystemExit("No hashed rows parsed.")

    return raw, rows


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


def infer_widths(headers: list[str], rows: list[list[object]], padding: int = 0) -> list[int]:
    widths = []
    for idx, header in enumerate(headers):
        values = [str(header)]
        values.extend(str(row[idx]) for row in rows)
        widths.append(max(len(v) for v in values) + padding)
    return widths


def write_fixed_section(f, title: str, meta_lines: list[str], headers: list[str], rows: list[list[object]], widths: list[int]) -> None:
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


def group_by_config(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row.bits, row.num_vars), []).append(row)
    return dict(sorted(grouped.items()))


def section_label(prefix: str, bits: int, num_vars: int) -> str:
    return f"{prefix} / bits={bits} / num_vars={num_vars}"


def write_text_report(path: Path, unhashed: list[UnhashedRow], hashed: list[HashedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write("Global Full-Montgomery SumCheck Benchmark\n")
        f.write("=========================================\n\n")

        unhashed_headers = [
            "template", "function", "N", "rows", "terms", "deg",
            "generic_ms", "spec_ms", "speedup", "generic_p90", "spec_p90",
            "generic_Mpts/s", "spec_Mpts/s", "shape",
        ]

        for (bits, num_vars), group in group_by_config(unhashed).items():
            body = [
                [
                    r.template, r.function, r.N, r.rows, r.terms, r.deg,
                    f"{r.generic_ms:.3f}", f"{r.spec_ms:.3f}", f"{r.speedup:.2f}x",
                    f"{r.generic_p90:.3f}", f"{r.spec_p90:.3f}",
                    f"{r.generic_mpts_s:.2f}", f"{r.spec_mpts_s:.2f}", r.shape,
                ]
                for r in group
            ]
            write_fixed_section(
                f,
                section_label("Unhashed / Paper Compare", bits, num_vars),
                [
                    f"bits: {bits}",
                    f"num_vars: {num_vars}",
                    f"device: {group[0].device}",
                    f"backend: {group[0].backend}",
                ],
                unhashed_headers,
                body,
                infer_widths(unhashed_headers, body),
            )

        if hashed:
            hashed_headers = [
                "template", "function", "N", "rows", "terms", "deg",
                "generic_ms", "spec_ms", "speedup", "generic_p90", "spec_p90",
                "generic_Mpts/s", "spec_Mpts/s", "sha3_prefix", "shape",
            ]

            for (bits, num_vars), group in group_by_config(hashed).items():
                body = [
                    [
                        r.template, r.function, r.N, r.rows, r.terms, r.deg,
                        f"{r.generic_ms:.3f}", f"{r.spec_ms:.3f}", f"{r.speedup:.2f}x",
                        f"{r.generic_p90:.3f}", f"{r.spec_p90:.3f}",
                        f"{r.generic_mpts_s:.2f}", f"{r.spec_mpts_s:.2f}",
                        r.transcript_prefix, r.shape,
                    ]
                    for r in group
                ]
                write_fixed_section(
                    f,
                    section_label("Hashed / SHA3 Transcript", bits, num_vars),
                    [
                        f"bits: {bits}",
                        f"num_vars: {num_vars}",
                        f"device: {group[0].device}",
                        f"backend: {group[0].backend}",
                    ],
                    hashed_headers,
                    body,
                    infer_widths(hashed_headers, body),
                )


def md_escape(x: object) -> str:
    return str(x).replace("|", "\\|")


def write_table_md(f, columns: list[str], rows: list[dict], labels: dict[str, str] | None = None) -> None:
    labels = labels or {}
    f.write("| " + " | ".join(labels.get(c, c) for c in columns) + " |\n")
    f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            vals.append(f"{val:.3f}" if isinstance(val, float) else md_escape(val))
        f.write("| " + " | ".join(vals) + " |\n")
    f.write("\n")


def write_md(path: Path, unhashed: list[UnhashedRow], hashed: list[HashedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write("# Global Full-Montgomery SumCheck Benchmark\n\n")

        for (bits, num_vars), group in group_by_config(unhashed).items():
            f.write(f"## Unhashed / Paper Compare / bits={bits} / num_vars={num_vars}\n\n")
            f.write(f"- bits: `{bits}`\n")
            f.write(f"- num_vars: `{num_vars}`\n")
            f.write(f"- device: `{group[0].device}`\n")
            f.write(f"- backend: `{group[0].backend}`\n\n")
            write_table_md(
                f,
                [
                    "template", "function", "N", "rows", "terms", "deg",
                    "generic_ms", "spec_ms", "speedup", "generic_p90",
                    "spec_p90", "generic_mpts_s", "spec_mpts_s", "shape",
                ],
                [asdict(r) for r in group],
                {"generic_mpts_s": "generic_Mpts/s", "spec_mpts_s": "spec_Mpts/s"},
            )

        if hashed:
            for (bits, num_vars), group in group_by_config(hashed).items():
                f.write(f"## Hashed / SHA3 Transcript / bits={bits} / num_vars={num_vars}\n\n")
                f.write(f"- bits: `{bits}`\n")
                f.write(f"- num_vars: `{num_vars}`\n")
                f.write(f"- device: `{group[0].device}`\n")
                f.write(f"- backend: `{group[0].backend}`\n\n")
                write_table_md(
                    f,
                    [
                        "template", "function", "N", "rows", "terms", "deg",
                        "generic_ms", "spec_ms", "speedup", "generic_p90", "spec_p90",
                        "generic_mpts_s", "spec_mpts_s", "transcript_prefix", "shape",
                    ],
                    [asdict(r) for r in group],
                    {
                        "generic_mpts_s": "generic_Mpts/s",
                        "spec_mpts_s": "spec_Mpts/s",
                        "transcript_prefix": "sha3_prefix",
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
    lines = ["# Global Benchmark Summary\n\n"]

    for (bits, num_vars), group in group_by_config(unhashed).items():
        speedups = [r.speedup for r in group]
        best = max(group, key=lambda r: r.speedup)
        worst = min(group, key=lambda r: r.speedup)
        lines += [
            f"## Unhashed / bits={bits} / num_vars={num_vars}\n\n",
            f"- rows: `{len(group)}`\n",
            f"- mean speedup: `{statistics.mean(speedups):.3f}x`\n",
            f"- median speedup: `{statistics.median(speedups):.3f}x`\n",
            f"- best speedup: `{best.template}` `{best.speedup:.3f}x`\n",
            f"- weakest speedup: `{worst.template}` `{worst.speedup:.3f}x`\n\n",
        ]

    if hashed:
        for (bits, num_vars), group in group_by_config(hashed).items():
            speedups = [r.speedup for r in group]
            best = max(group, key=lambda r: r.speedup)
            worst = min(group, key=lambda r: r.speedup)
            lines += [
                f"## Hashed / bits={bits} / num_vars={num_vars}\n\n",
                f"- rows: `{len(group)}`\n",
                f"- mean speedup: `{statistics.mean(speedups):.3f}x`\n",
                f"- median speedup: `{statistics.median(speedups):.3f}x`\n",
                f"- best speedup: `{best.template}` `{best.speedup:.3f}x`\n",
                f"- weakest speedup: `{worst.template}` `{worst.speedup:.3f}x`\n\n",
            ]

    path.write_text("".join(lines))



def median_runtime_pair(rows) -> tuple[float, float]:
    generic = [r.generic_ms for r in rows]
    spec = [r.spec_ms for r in rows]
    if not generic or not spec:
        return 0.0, 0.0
    return statistics.median(generic), statistics.median(spec)


def clear_old_plots(path: Path) -> None:
    """Keep report plots intentional by removing stale benchmark plots first."""
    if not path.exists():
        return
    for pattern in [
        "unhashed_runtime_bits*.png",
        "unhashed_speedup_bits*.png",
        "hashed_runtime_bits*.png",
        "scaling_*.png",
        "specialized_functions_bits*.png",
        "representative_scaling_*.png",
        "generic_vs_specialized_bits*.png",
    ]:
        for old in path.glob(pattern):
            old.unlink()


def plot_benchmark_figures(path: Path, unhashed: list[UnhashedRow], hashed: list[HashedRow]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"plots skipped: {e}")
        return

    path.mkdir(parents=True, exist_ok=True)
    clear_old_plots(path)

    standard_bits = PLOT_STANDARD_BITS
    standard_num_vars = PLOT_STANDARD_NUM_VARS
    rep_polys = PLOT_REPRESENTATIVE_POLYS
    plot_num_vars_values = PLOT_NUM_VARS_VALUES

    standard_rows = [
        r for r in unhashed
        if r.bits == standard_bits and r.num_vars == standard_num_vars
    ]

    # 1) Function comparison at medium size: specialized only.
    if standard_rows:
        ordered = sorted(standard_rows, key=lambda r: r.spec_ms)
        x = list(range(len(ordered)))
        labels = [r.template for r in ordered]

        plt.figure(figsize=(13, 5.5))
        plt.bar(x, [r.spec_ms for r in ordered])
        plt.xticks(x, labels, rotation=55, ha="right")
        plt.ylabel("specialized runtime (ms)")
        plt.title(
            f"Specialized runtime by function / bits={standard_bits} / num_vars={standard_num_vars}"
        )
        plt.tight_layout()
        plt.savefig(
            path / f"specialized_functions_bits{standard_bits}_nv{standard_num_vars}.png",
            dpi=160,
        )
        plt.close()

    # 2) Representative functions scaling over bit size at standard num_vars.
    bits_values = sorted({
        r.bits for r in unhashed
        if r.num_vars == standard_num_vars and r.template in rep_polys
    })

    if len(bits_values) >= 2:
        plt.figure(figsize=(8.5, 5.2))
        plotted = False

        for template in rep_polys:
            xs = []
            ys = []
            for bits in bits_values:
                match = next(
                    (
                        r for r in unhashed
                        if r.bits == bits
                        and r.num_vars == standard_num_vars
                        and r.template == template
                    ),
                    None,
                )
                if match is None:
                    continue
                xs.append(bits)
                ys.append(match.spec_ms)

            if len(xs) >= 2:
                plt.plot(xs, ys, marker="o", label=template)
                plotted = True

        if plotted:
            plt.xlabel("field bit width")
            plt.ylabel("specialized runtime (ms)")
            plt.title(f"Representative specialized scaling vs bit width / num_vars={standard_num_vars}")
            plt.xticks(bits_values)
            plt.legend(fontsize="small")
            plt.tight_layout()
            plt.savefig(
                path / f"representative_scaling_bits_nv{standard_num_vars}.png",
                dpi=160,
            )
        plt.close()

    # 3) Representative functions scaling over num_vars at standard bit width.
    available_num_vars = sorted({
        r.num_vars for r in unhashed
        if r.bits == standard_bits and r.template in rep_polys
    })
    selected_num_vars = [nv for nv in plot_num_vars_values if nv in available_num_vars]
    if len(selected_num_vars) < 2:
        selected_num_vars = available_num_vars

    if len(selected_num_vars) >= 2:
        plt.figure(figsize=(8.5, 5.2))
        plotted = False

        for template in rep_polys:
            xs = []
            ys = []
            for num_vars in selected_num_vars:
                match = next(
                    (
                        r for r in unhashed
                        if r.bits == standard_bits
                        and r.num_vars == num_vars
                        and r.template == template
                    ),
                    None,
                )
                if match is None:
                    continue
                xs.append(num_vars)
                ys.append(match.spec_ms)

            if len(xs) >= 2:
                plt.plot(xs, ys, marker="o", label=template)
                plotted = True

        if plotted:
            plt.xlabel("num_vars")
            plt.ylabel("specialized runtime (ms)")
            plt.title(f"Representative specialized scaling vs num_vars / bits={standard_bits}")
            plt.xticks(selected_num_vars)
            plt.legend(fontsize="small")
            plt.tight_layout()
            plt.savefig(
                path / f"representative_scaling_num_vars_bits{standard_bits}.png",
                dpi=160,
            )
        plt.close()

    # 4) Generic vs specialized runtime across all tests at median size.
    if standard_rows:
        ordered = sorted(standard_rows, key=lambda r: r.template)
        x = list(range(len(ordered)))
        width = 0.38
        labels = [r.template for r in ordered]

        plt.figure(figsize=(14, 5.8))
        plt.bar([i - width / 2 for i in x], [r.generic_ms for r in ordered], width, label="generic")
        plt.bar([i + width / 2 for i in x], [r.spec_ms for r in ordered], width, label="specialized")
        plt.xticks(x, labels, rotation=55, ha="right")
        plt.ylabel("runtime (ms)")
        plt.title(f"Generic vs specialized runtime / bits={standard_bits} / num_vars={standard_num_vars}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            path / f"generic_vs_specialized_bits{standard_bits}_nv{standard_num_vars}.png",
            dpi=160,
        )
        plt.close()


def main() -> None:
    global PLOT_STANDARD_BITS, PLOT_STANDARD_NUM_VARS
    global PLOT_NUM_VARS_VALUES, PLOT_REPRESENTATIVE_POLYS

    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="64", help="32, 64, 128, comma-list, or all")
    ap.add_argument("--num-vars", default="20", help="single value or comma-list")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--polys", default=",".join(DEFAULT_POLYS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--include-hashed", action="store_true", help="Run u64 SHA3 transcript benchmark section")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--plot-standard-bits", type=int, default=PLOT_STANDARD_BITS)
    ap.add_argument("--plot-standard-num-vars", type=int, default=PLOT_STANDARD_NUM_VARS)
    ap.add_argument("--plot-num-vars-values", default=",".join(str(x) for x in PLOT_NUM_VARS_VALUES))
    ap.add_argument("--plot-polys", default=",".join(PLOT_REPRESENTATIVE_POLYS))
    args = ap.parse_args()

    PLOT_STANDARD_BITS = int(args.plot_standard_bits)
    PLOT_STANDARD_NUM_VARS = int(args.plot_standard_num_vars)
    PLOT_NUM_VARS_VALUES = [4, 16, 20]
    PLOT_REPRESENTATIVE_POLYS = [
        x.strip()
        for x in str(args.plot_polys).split(",")
        if x.strip()
    ]


    raw_dir = args.out_dir / "raw"
    plot_dir = args.out_dir / "plots"
    raw_dir.mkdir(parents=True, exist_ok=True)

    unhashed_raw, unhashed = run_unhashed(args, args.polys)
    safe_bits = str(args.bits).replace(",", "_")
    safe_num_vars = str(args.num_vars).replace(",", "_")
    (raw_dir / f"global_unhashed_bits{safe_bits}_nv{safe_num_vars}.txt").write_text(unhashed_raw)

    hashed: list[HashedRow] = []
    if args.include_hashed:
        hashed_raw, hashed = run_hashed(args, args.polys)
        (raw_dir / f"global_hashed_bits64_nv{safe_num_vars}.txt").write_text(hashed_raw)

    write_csv(args.out_dir / "global_benchmark_unhashed.csv", unhashed)
    if hashed:
        write_csv(args.out_dir / "global_benchmark_hashed.csv", hashed)

    write_md(args.out_dir / "global_benchmark.md", unhashed, hashed)
    write_text_report(args.out_dir / "global_benchmark.txt", unhashed, hashed)
    write_summary(args.out_dir / "global_benchmark_summary.md", unhashed, hashed)

    if not args.no_plot:
        plot_benchmark_figures(plot_dir, unhashed, hashed)

    print(f"wrote markdown: {args.out_dir / 'global_benchmark.md'}")
    print(f"wrote text:     {args.out_dir / 'global_benchmark.txt'}")
    print(f"wrote summary:  {args.out_dir / 'global_benchmark_summary.md'}")
    print(f"wrote csv:      {args.out_dir / 'global_benchmark_unhashed.csv'}")
    if hashed:
        print(f"wrote hashed:   {args.out_dir / 'global_benchmark_hashed.csv'}")
    if not args.no_plot:
        print(f"wrote plots:    {plot_dir}")


if __name__ == "__main__":
    main()
