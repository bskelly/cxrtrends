#!/usr/bin/env python3
"""Create a compact tabular report from custom CXR experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_seed_results(results_root: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(results_root.glob("*/seed_results.csv")):
        experiment_label = csv_path.parent.name
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df.insert(0, "experiment_label", experiment_label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_experiment_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return seed_df
    return (
        seed_df.groupby(["experiment_label", "task"], dropna=False)
        .agg(
            mean_metric=("metric", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias=("bias", "mean"),
            mean_wape=("wape", "mean"),
            mean_smape=("smape", "mean"),
            seeds=("seed", "count"),
        )
        .reset_index()
        .sort_values(["experiment_label", "task"], kind="mergesort")
    )


def build_markdown_report(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "# CXR Results Report\n\nNo experiment summaries were found.\n"

    lines = [
        "# CXR Results Report",
        "",
        "| Experiment | Task | MAE | RMSE | Bias | WAPE | sMAPE | Seeds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| {row.experiment_label} | {row.task} | "
            f"{row.mean_mae:.3f} | {row.mean_rmse:.3f} | {row.mean_bias:.3f} | "
            f"{row.mean_wape:.3f} | {row.mean_smape:.3f} | {row.seeds} |"
        )
    lines.append("")
    lines.append(
        "Lower is better for MAE, RMSE, WAPE, and sMAPE. Bias near zero indicates less systematic over- or under-forecasting."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact report from CXR experiment outputs.")
    parser.add_argument(
        "--results-root",
        required=True,
        help="Path to an experiment output folder, e.g. benchmark_results/cxr_compare",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists():
        parser.exit(status=1, message=f"Error: results root does not exist: {results_root}\n")

    seed_df = load_seed_results(results_root)
    summary_df = build_experiment_summary(seed_df)

    seed_path = results_root / "report_seed_results.csv"
    summary_path = results_root / "report_summary.csv"
    markdown_path = results_root / "report.md"

    seed_df.to_csv(seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    markdown_path.write_text(build_markdown_report(summary_df), encoding="utf-8")

    print(f"Wrote {seed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
