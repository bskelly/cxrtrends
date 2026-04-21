#!/usr/bin/env python3
"""Small smoke test for the Phase 2 CXR daily forecast task."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cik_benchmark.tasks.cxr_daily_volume_tasks import CXRDailyForecastTask


def main() -> int:
    task = CXRDailyForecastTask(
        seed=7,
        fixed_config={
            "normalized_path": str(
                PROJECT_ROOT / "data" / "normalized" / "cxr_trends.csv"
            ),
            "frequency": "D",
            "horizon": 7,
            "history_length": 28,
            "age_upper_years": 1.0,
            "series_mode": "panel",
            "panel_column": "facility_id",
            "entity_id": "CHI",
            "context_text": "Example placeholder context. Replace this with clinical text later.",
        },
    )

    print(f"Task: {task.name}")
    print(f"Series ID: {task.series_id}")
    print(f"Past shape: {task.past_time.shape}")
    print(f"Future shape: {task.future_time.shape}")
    print(f"Past start: {task.past_time.index.min().date()}")
    print(f"Future end: {task.future_time.index.max().date()}")
    print(f"Scenario: {task.scenario}")
    print(task.past_time.head(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
