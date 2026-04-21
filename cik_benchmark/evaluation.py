"""Minimal evaluation utilities for custom CXR experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from cik_benchmark.metrics import forecast_table_from_samples


def evaluate_task_instance(task_cls, seed, method_callable, n_samples, task_config, output_folder):
    task = task_cls(seed=seed, fixed_config=task_config)
    samples, extra_info = method_callable(task_instance=task, n_samples=n_samples)
    evaluation = task.evaluate(samples)
    forecast_table = forecast_table_from_samples(task.future_time, samples)

    seed_folder = output_folder / task_cls.__name__ / f"seed_{seed}"
    seed_folder.mkdir(parents=True, exist_ok=True)

    task.past_time.to_csv(seed_folder / "past_time.csv", index=True)
    task.future_time.to_csv(seed_folder / "future_time.csv", index=True)
    pd.DataFrame(samples[:, :, 0]).to_csv(seed_folder / "samples.csv", index=False)
    forecast_table.to_csv(seed_folder / "forecast_table.csv", index=False)
    (seed_folder / "context.txt").write_text(
        f"Background:\n{task.background}\n\nScenario:\n{task.scenario}\n",
        encoding="utf-8",
    )
    (seed_folder / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (seed_folder / "extra_info.json").write_text(
        json.dumps(extra_info, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return {
        "task": task_cls.__name__,
        "seed": seed,
        "series_id": getattr(task, "series_id", None),
        **evaluation,
    }


def summarize_results(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df

    summary = (
        df.groupby("task", dropna=False)
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
    )
    return df, summary
