from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from cik_benchmark.baselines.cxr_baselines import (
    DirectContextPromptBaseline,
    LastValueBaseline,
)
from cik_benchmark.evaluation import evaluate_task_instance
from cik_benchmark.tasks.cxr_daily_volume_tasks import CXRDailyForecastUnder1YearTask
from tests.helpers import RSV_CONTEXT, TempDirTestCaseMixin, write_normalized_csv


class TestForecastingSmoke(TempDirTestCaseMixin, unittest.TestCase):
    def test_last_value_smoke_run_writes_outputs(self):
        normalized_path = write_normalized_csv(self.tmp_path / "normalized.csv", days=80)
        output_folder = self.tmp_path / "outputs"

        row = evaluate_task_instance(
            task_cls=CXRDailyForecastUnder1YearTask,
            seed=3,
            method_callable=LastValueBaseline(use_context=False),
            n_samples=3,
            task_config={
                "normalized_path": str(normalized_path),
                "frequency": "D",
                "horizon": 7,
                "history_length": 28,
                "series_mode": "single",
            },
            output_folder=output_folder,
        )

        self.assertIn("mae", row)
        forecast_table = (
            output_folder
            / "CXRDailyForecastUnder1YearTask"
            / "seed_3"
            / "forecast_table.csv"
        )
        self.assertTrue(forecast_table.exists())

    def test_direct_context_prompt_can_change_output_with_rsv_text(self):
        normalized_path = write_normalized_csv(self.tmp_path / "normalized.csv", days=80)
        base_config = {
            "normalized_path": str(normalized_path),
            "frequency": "D",
            "horizon": 7,
            "history_length": 28,
            "series_mode": "single",
        }

        task_no_context = CXRDailyForecastUnder1YearTask(seed=1, fixed_config=base_config)
        task_with_context = CXRDailyForecastUnder1YearTask(
            seed=1,
            fixed_config={**base_config, "context_text": RSV_CONTEXT},
        )

        def fake_chat_completion(self, prompt: str, n_samples: int):
            future_dates = task_with_context.future_time.index
            value = 2 if "Nirsevimab" in prompt else 6
            payload = {
                "forecast": [
                    {"date": idx.strftime("%Y-%m-%d"), "value": value}
                    for idx in future_dates
                ]
            }
            return {
                "choices": [
                    {"message": {"content": json.dumps(payload)}} for _ in range(n_samples)
                ],
                "usage": {},
            }

        with patch.object(
            DirectContextPromptBaseline, "_chat_completion", new=fake_chat_completion
        ):
            baseline_no_context = DirectContextPromptBaseline(
                use_context=False, api_key="dummy-key"
            )
            baseline_with_context = DirectContextPromptBaseline(
                use_context=True, api_key="dummy-key"
            )
            no_context_samples, _ = baseline_no_context(task_no_context, n_samples=2)
            with_context_samples, extra = baseline_with_context(
                task_with_context, n_samples=2
            )

        self.assertTrue((no_context_samples[:, :, 0] == 6).all())
        self.assertTrue((with_context_samples[:, :, 0] == 2).all())
        self.assertIn("Nirsevimab", extra["prompt"])


if __name__ == "__main__":
    unittest.main()
