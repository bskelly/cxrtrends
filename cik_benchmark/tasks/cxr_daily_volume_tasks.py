"""Custom daily forecasting tasks built on the normalized CXR trends dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..base import UnivariateCRPSTask
from ..data.cxr_daily import CXRDailyVolumeAdapter
from ..metrics import point_forecast_metrics
from . import WeightCluster


DEFAULT_NORMALIZED_DATASET = (
    Path(__file__).resolve().parents[2] / "data" / "normalized" / "cxr_trends.csv"
)


class CXRDailyForecastTask(UnivariateCRPSTask):
    """Generic daily exam-volume forecasting task for the CXR trends dataset."""

    _context_sources = UnivariateCRPSTask._context_sources + ["c_h", "c_i"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.1"

    def __init__(self, seed: int | None = None, fixed_config: Optional[dict] = None):
        self.fixed_config = fixed_config or {}
        self.normalized_path = self.fixed_config.get(
            "normalized_path", str(DEFAULT_NORMALIZED_DATASET)
        )
        self.frequency = self.fixed_config.get("frequency", "D")
        raw_horizon = self.fixed_config.get("horizon", 7)
        self.horizon = None if raw_horizon is None else int(raw_horizon)
        raw_history_length = self.fixed_config.get("history_length", 56)
        self.history_length = (
            None if raw_history_length is None else int(raw_history_length)
        )
        self.age_upper_years = self.fixed_config.get("age_upper_years")
        self.series_mode = self.fixed_config.get("series_mode", "single")
        self.panel_column = self.fixed_config.get("panel_column")
        self.entity_id = self.fixed_config.get("entity_id")
        self.context_text = self.fixed_config.get("context_text")
        self.context_builder = self.fixed_config.get("context_builder")
        self.context_source_column = self.fixed_config.get(
            "context_source_column", "exam_reason_text"
        )
        self.forecast_start_date = self.fixed_config.get("forecast_start_date")
        self.forecast_end_date = self.fixed_config.get("forecast_end_date")
        self.series_id = None
        self.task_metadata = {}
        super().__init__(seed=seed, fixed_config=None)

    def random_instance(self) -> None:
        adapter = CXRDailyVolumeAdapter(
            normalized_path=self.normalized_path,
            frequency=self.frequency,
            horizon=self.horizon,
            history_length=self.history_length,
            age_upper_years=self.age_upper_years,
            series_mode=self.series_mode,
            panel_column=self.panel_column,
            entity_id=self.entity_id,
            context_text=self.context_text,
            context_builder=self.context_builder,
            context_source_column=self.context_source_column,
            forecast_start_date=self.forecast_start_date,
            forecast_end_date=self.forecast_end_date,
        )
        window = adapter.sample_window(self.random)
        self.series_id = window.series_id
        self.task_metadata = window.metadata
        self.past_time = window.past_time
        self.future_time = window.future_time
        self.constraints = None
        self.background = window.background
        self.scenario = window.scenario
        self.region_of_interest = None

    def verify_config(self) -> list[str]:
        errors = super().verify_config()
        if self.frequency != "D":
            errors.append("This custom task currently supports daily frequency only.")
        if list(self.past_time.columns) != ["exam_count"]:
            errors.append("Task expects a single target column named 'exam_count'.")
        return errors

    def evaluate(self, samples):
        target = self.future_time["exam_count"].to_numpy(dtype=float)
        return point_forecast_metrics(target=target, samples=samples)


class CXRDailyForecastUnder6MonthsTask(CXRDailyForecastTask):
    """Convenience task for exams under 6 months."""

    __version__ = "0.0.1"

    def __init__(self, seed: int | None = None, fixed_config: Optional[dict] = None):
        config = dict(fixed_config or {})
        config.setdefault("age_upper_years", 0.5)
        super().__init__(seed=seed, fixed_config=config)


class CXRDailyForecastUnder1YearTask(CXRDailyForecastTask):
    """Convenience task for exams under 1 year."""

    __version__ = "0.0.1"

    def __init__(self, seed: int | None = None, fixed_config: Optional[dict] = None):
        config = dict(fixed_config or {})
        config.setdefault("age_upper_years", 1.0)
        super().__init__(seed=seed, fixed_config=config)


__TASKS__ = [
    CXRDailyForecastTask,
    CXRDailyForecastUnder6MonthsTask,
    CXRDailyForecastUnder1YearTask,
]

__CLUSTERS__ = [
    WeightCluster(weight=1, tasks=__TASKS__),
]
