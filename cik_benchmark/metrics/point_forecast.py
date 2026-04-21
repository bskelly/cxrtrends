"""Simple point-forecast metrics for count forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_point_forecast(samples) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim == 3:
        return np.median(samples[:, :, 0], axis=0).astype(float)
    if samples.ndim == 2:
        return np.median(samples, axis=0).astype(float)
    raise RuntimeError(
        "samples must have shape (n_samples, horizon, 1) or (n_samples, horizon)"
    )


def point_forecast_metrics(target, samples) -> dict[str, float]:
    """Compute compact tabular metrics from forecast samples.

    We evaluate the sample median as the point forecast. This is a pragmatic
    local metric layer for the custom task family and does not attempt to
    reproduce the full original benchmark's distributional scoring.
    """

    point_forecast = _to_point_forecast(samples)
    target = np.asarray(target, dtype=float)
    errors = point_forecast - target
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    bias = float(np.mean(errors))
    total_actual = float(np.sum(np.abs(target)))
    wape = float(np.sum(abs_errors) / total_actual) if total_actual > 0 else float("nan")

    denom = np.abs(point_forecast) + np.abs(target)
    smape_terms = np.where(denom == 0, 0.0, 2.0 * abs_errors / denom)
    smape = float(np.mean(smape_terms))

    return {
        "metric": mae,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "wape": wape,
        "smape": smape,
    }


def forecast_table_from_samples(future_time: pd.DataFrame, samples) -> pd.DataFrame:
    """Build a simple per-date forecast inspection table."""

    point_forecast = _to_point_forecast(samples)
    samples_arr = np.asarray(samples)
    if samples_arr.ndim == 3:
        scalar_samples = samples_arr[:, :, 0]
    else:
        scalar_samples = samples_arr

    table = pd.DataFrame(
        {
            "date": future_time.index,
            "actual": future_time.iloc[:, 0].to_numpy(dtype=float),
            "forecast_p50": point_forecast,
            "forecast_mean": np.mean(scalar_samples, axis=0).astype(float),
            "forecast_p10": np.quantile(scalar_samples, 0.10, axis=0).astype(float),
            "forecast_p90": np.quantile(scalar_samples, 0.90, axis=0).astype(float),
        }
    )
    table["error"] = table["forecast_p50"] - table["actual"]
    table["abs_error"] = np.abs(table["error"])
    return table
