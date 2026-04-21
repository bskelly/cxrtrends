"""Minimal evaluation utilities for custom CXR experiments."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from cik_benchmark.metrics import forecast_table_from_samples


SVG_WIDTH = 1100
SVG_HEIGHT = 420
PLOT_LEFT = 70
PLOT_RIGHT = 30
PLOT_TOP = 40
PLOT_BOTTOM = 70


def _scale_x(index_len: int) -> np.ndarray:
    if index_len <= 1:
        return np.array([PLOT_LEFT + (SVG_WIDTH - PLOT_LEFT - PLOT_RIGHT) / 2.0])
    usable_width = SVG_WIDTH - PLOT_LEFT - PLOT_RIGHT
    return np.linspace(PLOT_LEFT, PLOT_LEFT + usable_width, index_len)


def _scale_y(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    usable_height = SVG_HEIGHT - PLOT_TOP - PLOT_BOTTOM
    if np.isclose(min_val, max_val):
        return np.full(values.shape, PLOT_TOP + usable_height / 2.0)
    normalized = (values - min_val) / (max_val - min_val)
    return PLOT_TOP + usable_height * (1.0 - normalized)


def _points(xs: np.ndarray, ys: np.ndarray) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=False))


def save_forecast_plot(task, samples, path: Path) -> None:
    samples_arr = np.asarray(samples, dtype=float)
    scalar_samples = samples_arr[:, :, 0] if samples_arr.ndim == 3 else samples_arr

    past_values = task.past_time.iloc[:, 0].to_numpy(dtype=float)
    future_values = task.future_time.iloc[:, 0].to_numpy(dtype=float)
    p10 = np.quantile(scalar_samples, 0.10, axis=0).astype(float)
    p50 = np.quantile(scalar_samples, 0.50, axis=0).astype(float)
    p90 = np.quantile(scalar_samples, 0.90, axis=0).astype(float)

    all_values = np.concatenate([past_values, future_values, p10, p50, p90])
    min_val = float(np.nanmin(all_values))
    max_val = float(np.nanmax(all_values))
    if np.isclose(min_val, max_val):
        min_val -= 1.0
        max_val += 1.0

    total_len = len(past_values) + len(future_values)
    x_all = _scale_x(total_len)
    x_past = x_all[: len(past_values)]
    x_future = x_all[len(past_values) :]

    y_past = _scale_y(past_values, min_val, max_val)
    y_future = _scale_y(future_values, min_val, max_val)
    y_p10 = _scale_y(p10, min_val, max_val)
    y_p50 = _scale_y(p50, min_val, max_val)
    y_p90 = _scale_y(p90, min_val, max_val)

    divider_x = x_future[0] if len(x_future) else x_all[-1]
    band_points = _points(
        np.concatenate([x_future, x_future[::-1]]),
        np.concatenate([y_p10, y_p90[::-1]]),
    )

    title = task.name if task.scenario is None else f"{task.name} | {task.scenario[:120]}"
    x_start = str(task.future_time.index.min().date()) if len(task.future_time.index) else "n/a"
    x_end = str(task.future_time.index.max().date()) if len(task.future_time.index) else "n/a"

    y_ticks = np.linspace(min_val, max_val, 5)
    y_tick_lines = []
    for tick in y_ticks:
        y = float(_scale_y(np.array([tick]), min_val, max_val)[0])
        y_tick_lines.append(
            f'<line x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{SVG_WIDTH - PLOT_RIGHT}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        y_tick_lines.append(
            f'<text x="{PLOT_LEFT - 10}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="#374151">{tick:.1f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{PLOT_LEFT}" y="24" font-size="18" font-weight="600" fill="#111827">{escape(title)}</text>
  <text x="{PLOT_LEFT}" y="{SVG_HEIGHT - 18}" font-size="12" fill="#4b5563">Forecast window: {escape(x_start)} to {escape(x_end)}</text>
  {''.join(y_tick_lines)}
  <line x1="{PLOT_LEFT}" y1="{SVG_HEIGHT - PLOT_BOTTOM}" x2="{SVG_WIDTH - PLOT_RIGHT}" y2="{SVG_HEIGHT - PLOT_BOTTOM}" stroke="#111827" stroke-width="1.2" />
  <line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{SVG_HEIGHT - PLOT_BOTTOM}" stroke="#111827" stroke-width="1.2" />
  <line x1="{divider_x:.2f}" y1="{PLOT_TOP}" x2="{divider_x:.2f}" y2="{SVG_HEIGHT - PLOT_BOTTOM}" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="5 4" />
  <polygon points="{band_points}" fill="#93c5fd" opacity="0.45" />
  <polyline points="{_points(x_past, y_past)}" fill="none" stroke="#111827" stroke-width="2" />
  <polyline points="{_points(x_future, y_future)}" fill="none" stroke="#f59e0b" stroke-width="2" />
  <polyline points="{_points(x_future, y_p50)}" fill="none" stroke="#2563eb" stroke-width="2" />
  <circle cx="{PLOT_LEFT + 12}" cy="{SVG_HEIGHT - 48}" r="4" fill="#111827" />
  <text x="{PLOT_LEFT + 24}" y="{SVG_HEIGHT - 44}" font-size="12" fill="#111827">History</text>
  <circle cx="{PLOT_LEFT + 110}" cy="{SVG_HEIGHT - 48}" r="4" fill="#f59e0b" />
  <text x="{PLOT_LEFT + 122}" y="{SVG_HEIGHT - 44}" font-size="12" fill="#111827">Actual future</text>
  <circle cx="{PLOT_LEFT + 240}" cy="{SVG_HEIGHT - 48}" r="4" fill="#2563eb" />
  <text x="{PLOT_LEFT + 252}" y="{SVG_HEIGHT - 44}" font-size="12" fill="#111827">Forecast median</text>
  <rect x="{PLOT_LEFT + 386}" y="{SVG_HEIGHT - 54}" width="10" height="10" fill="#93c5fd" opacity="0.45" />
  <text x="{PLOT_LEFT + 402}" y="{SVG_HEIGHT - 44}" font-size="12" fill="#111827">80% interval</text>
</svg>
'''

    (path / "forecast_plot.svg").write_text(svg, encoding="utf-8")


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
    save_forecast_plot(task=task, samples=samples, path=seed_folder)
    context_text = (
        f"Background:\n{task.background}\n\n"
        f"Scenario:\n{task.scenario}\n"
    )
    (seed_folder / "context.txt").write_text(context_text, encoding="utf-8")
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
