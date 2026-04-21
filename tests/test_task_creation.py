from __future__ import annotations

import unittest

from cik_benchmark.data.cxr_trends import ValidationError
from cik_benchmark.tasks.cxr_daily_volume_tasks import CXRDailyForecastUnder1YearTask
from tests.helpers import RSV_CONTEXT, TempDirTestCaseMixin, write_normalized_csv


class TestTaskCreation(TempDirTestCaseMixin, unittest.TestCase):
    def test_panel_task_creation_preserves_context(self):
        normalized_path = write_normalized_csv(self.tmp_path / "normalized.csv", days=70)

        task = CXRDailyForecastUnder1YearTask(
            seed=11,
            fixed_config={
                "normalized_path": str(normalized_path),
                "frequency": "D",
                "horizon": 7,
                "history_length": 28,
                "series_mode": "panel",
                "panel_column": "facility_id",
                "entity_id": "CHI",
                "context_text": RSV_CONTEXT,
            },
        )

        self.assertEqual(task.series_id, "CHI")
        self.assertEqual(task.past_time.shape, (28, 1))
        self.assertEqual(task.future_time.shape, (7, 1))
        self.assertEqual(task.scenario, RSV_CONTEXT)
        self.assertEqual(list(task.past_time.columns), ["exam_count"])

    def test_task_creation_fails_with_invalid_panel_column(self):
        normalized_path = write_normalized_csv(self.tmp_path / "normalized.csv", days=70)

        with self.assertRaises(ValidationError):
            CXRDailyForecastUnder1YearTask(
                seed=11,
                fixed_config={
                    "normalized_path": str(normalized_path),
                    "frequency": "D",
                    "horizon": 7,
                    "history_length": 28,
                    "series_mode": "panel",
                    "panel_column": "unknown_field",
                },
            )

    def test_fixed_start_date_uses_all_remaining_future_periods(self):
        normalized_path = write_normalized_csv(self.tmp_path / "normalized.csv", days=80)

        task = CXRDailyForecastUnder1YearTask(
            seed=11,
            fixed_config={
                "normalized_path": str(normalized_path),
                "frequency": "D",
                "horizon": None,
                "history_length": None,
                "series_mode": "single",
                "forecast_start_date": "2025-10-01",
            },
        )

        self.assertEqual(str(task.future_time.index.min().date()), "2025-10-01")
        self.assertEqual(str(task.past_time.index.max().date()), "2025-09-30")
        self.assertGreater(len(task.future_time), 1)


if __name__ == "__main__":
    unittest.main()
