#!/usr/bin/env python3
"""
Result Aggregation & Comparison Tool
Aggregates JSON results from all stages and generates comparison table.

Usage:
    python aggregate_results.py              # uses ./results by default
    python aggregate_results.py --save       # also writes comparison_report.txt
    python aggregate_results.py --dir PATH   # custom results directory
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

DEFAULT_RESULTS_BASE = Path("./results")

STAGE_LABELS = {
    1: "Stage 1 — Fluency  (lambada / wikitext / hellaswag)",
    2: "Stage 2 — Knowledge (VMLU / MMLU / Vietnamese tasks)",
    3: "Stage 3 — Reasoning (TruthfulQA / ARC / XCOPA)",
}

# Perplexity metrics (lower is better); everything else higher is better
PERPLEXITY_METRICS = {"perplexity,none", "word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none"}


def find_latest_result(stage_dir: Path) -> Path | None:
    """Find the most recent results_*.json under a stage directory."""
    files = sorted(stage_dir.rglob("results_*.json"))
    return files[-1] if files else None


def load_task_results(stage_dir: Path) -> dict:
    """Return the 'results' dict from the latest JSON in a stage directory."""
    path = find_latest_result(stage_dir)
    if path is None:
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("results", {})


def extract_metrics(task_results: dict) -> dict:
    """
    Flatten each task's metric dict.
    Keys like 'acc,none', 'acc_norm,none', 'perplexity,none', etc.
    """
    out = {}
    for task, metrics in task_results.items():
        if not isinstance(metrics, dict):
            continue
        out[task] = {k: v for k, v in metrics.items() if not k.endswith("_stderr,none") and k != "alias"}
    return out


def fmt(val, width=9):
    if val is None:
        return "  N/A    "
    return f"{val:{width}.4f}"


def delta_str(b, e):
    if b is None or e is None:
        return "   N/A  "
    d = e - b
    return f"{d:+.4f}"


def print_stage(stage_num: int, baseline_dir: Path, exp_dir: Path, lines: list[str]):
    header = STAGE_LABELS[stage_num]
    sep = "=" * 100

    def out(s=""):
        print(s)
        lines.append(s)

    out()
    out(sep)
    out(f"  {header}")
    out(sep)

    b_raw = load_task_results(baseline_dir / f"stage_{stage_num}")
    e_raw = load_task_results(exp_dir / f"stage_{stage_num}")

    if not b_raw and not e_raw:
        out(f"  ⚠  No results found for stage {stage_num}")
        return

    b_metrics = extract_metrics(b_raw)
    e_metrics = extract_metrics(e_raw)

    all_tasks = sorted(set(b_metrics) | set(e_metrics))

    # Collect all metric keys present across tasks
    all_keys = set()
    for t in all_tasks:
        all_keys |= set(b_metrics.get(t, {}).keys())
        all_keys |= set(e_metrics.get(t, {}).keys())

    # Focus on the most common ones
    priority = ["acc,none", "acc_norm,none", "perplexity,none", "word_perplexity,none", "bits_per_byte,none"]
    keys = [k for k in priority if k in all_keys]

    # Column header
    col_w = 11
    header_row = f"  {'Task':<35}"
    for k in keys:
        short = k.replace(",none", "").replace("_", " ")
        header_row += f"  {'Base':>{col_w}}  {'Exp':>{col_w}}  {'Δ':>{col_w}}"
    out(f"  {'Task':<35}" + "".join(f"  {'Base ':>11}  {'Exp ':>11}  {'Δ ':>9}  " for _ in keys))
    out(f"  {'':35}" + "".join(f"  {k.replace(',none','').replace('_',' '):>11}  {'':>11}  {'':>9}  " for k in keys))
    out("  " + "-" * 97)

    for task in all_tasks:
        row = f"  {task:<35}"
        for k in keys:
            b_val = b_metrics.get(task, {}).get(k)
            e_val = e_metrics.get(task, {}).get(k)
            lower_is_better = k in PERPLEXITY_METRICS
            row += f"  {fmt(b_val):>11}  {fmt(e_val):>11}  {delta_str(b_val, e_val):>9}  "
        out(row)

    out()


def main():
    parser = argparse.ArgumentParser(description="Aggregate distillation comparison results")
    parser.add_argument("--save", action="store_true", help="Write comparison_report.txt alongside results")
    parser.add_argument("--dir", default=str(DEFAULT_RESULTS_BASE), help="Results directory containing baseline and experimental subdirs")
    args = parser.parse_args()

    results_dir = Path(args.dir)

    # Auto-discover exactly two subdirectories: treat alphabetically first as baseline, second as experimental
    subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if len(subdirs) < 2:
        print(f"ERROR: Expected at least 2 model subdirectories in {results_dir}, found {len(subdirs)}")
        sys.exit(1)

    # Heuristic: baseline dir is the one NOT containing "skip" or "exp"
    def is_baseline(d: Path) -> bool:
        name = d.name.lower()
        return not any(x in name for x in ("skip", "exp", "800"))

    baselines = [d for d in subdirs if is_baseline(d)]
    experimentals = [d for d in subdirs if not is_baseline(d)]

    if not baselines or not experimentals:
        # Fall back to alphabetical order
        baselines = [subdirs[0]]
        experimentals = [subdirs[1]]

    baseline_dir = baselines[0]
    exp_dir = experimentals[0]

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    sep = "=" * 100
    out(sep)
    out(f"  1.7B Distillation Comparison Report")
    out(f"  Baseline   : {baseline_dir.name}")
    out(f"  Experimental: {exp_dir.name}")
    out(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out(sep)

    for stage in (1, 2, 3):
        print_stage(stage, baseline_dir, exp_dir, lines)

    out()
    out("Legend:")
    out("  Δ = Experimental − Baseline  (positive = experimental is better for acc/acc_norm)")
    out("  For perplexity metrics: negative Δ = experimental is better")

    if args.save:
        report_path = results_dir / "comparison_report.txt"
        report_path.write_text("\n".join(lines) + "\n")
        print(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    main()
