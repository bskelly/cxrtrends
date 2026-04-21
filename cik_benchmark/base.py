"""Lightweight task abstractions compatible with the original benchmark style."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


ALLOWED_CONTEXT_SOURCES = ["c_h", "c_i", "c_f", "c_cov", "c_causal"]
ALLOWED_SKILLS = [
    "forecasting",
    "natural language processing",
    "instruction following",
    "retrieval: context",
    "retrieval: memory",
    "reasoning: analogy",
    "reasoning: deduction",
    "reasoning: math",
    "reasoning: causal",
]


class BaseTask(ABC):
    """Base task contract matching the benchmark's past/future interface."""

    _context_sources = []
    _skills = ["forecasting", "natural language processing"]
    __version__ = "0.0.1"

    def __init__(self, seed: int | None = None, fixed_config: Optional[dict] = None):
        self.random = np.random.RandomState(seed)

        if fixed_config is not None:
            self.past_time = fixed_config["past_time"]
            self.future_time = fixed_config["future_time"]
            self.constraints = fixed_config.get("constraints")
            self.background = fixed_config.get("background")
            self.scenario = fixed_config.get("scenario")
        else:
            self.past_time = None
            self.future_time = None
            self.constraints = None
            self.background = None
            self.scenario = None
            self.random_instance()

        errors = self.verify_config()
        if errors:
            raise RuntimeError(
                f"Incorrect config for {self.__class__.__name__}: {errors}"
            )

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def verify_config(self) -> list[str]:
        errors = []
        if not isinstance(self.past_time, pd.DataFrame):
            errors.append("past_time must be a pandas DataFrame")
        if not isinstance(self.future_time, pd.DataFrame):
            errors.append("future_time must be a pandas DataFrame")
        for source in self._context_sources:
            if source not in ALLOWED_CONTEXT_SOURCES:
                errors.append(f"Invalid task context source: {source}")
        for skill in self._skills:
            if skill not in ALLOWED_SKILLS:
                errors.append(f"Invalid task skill: {skill}")
        if isinstance(self.past_time, pd.DataFrame) and isinstance(
            self.future_time, pd.DataFrame
        ):
            if list(self.past_time.columns) != list(self.future_time.columns):
                errors.append("past_time and future_time must share the same columns")
            if len(self.past_time) == 0:
                errors.append("past_time must contain at least one row")
            if len(self.future_time) == 0:
                errors.append("future_time must contain at least one row")
        return errors

    @abstractmethod
    def random_instance(self) -> None:
        """Populate the task with a random valid instance."""

    def evaluate(self, samples):
        raise RuntimeError(
            "Metrics integration is intentionally deferred for this custom task family."
        )


class UnivariateCRPSTask(BaseTask):
    """Compatibility class for univariate forecasting tasks."""

    __version__ = "0.0.1"

    def __init__(self, seed: int | None = None, fixed_config: Optional[dict] = None):
        if fixed_config is not None:
            self.region_of_interest = fixed_config.get("region_of_interest")
            self.roi_weight = fixed_config.get("roi_weight", 0.5)
            self.metric_constraint = fixed_config.get("metric_constraint")
        else:
            self.region_of_interest = None
            self.roi_weight = 0.5
            self.metric_constraint = None
        super().__init__(seed=seed, fixed_config=fixed_config)

    def verify_config(self) -> list[str]:
        errors = super().verify_config()
        if not 0 <= self.roi_weight <= 1:
            errors.append("roi_weight must be between 0 and 1")
        return errors
