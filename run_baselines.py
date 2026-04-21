"""Minimal experiment runner for the custom CXR task family."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from cik_benchmark.baselines import DirectContextPromptBaseline, LastValueBaseline
from cik_benchmark.evaluation import evaluate_task_instance, summarize_results
from cik_benchmark.tasks.cxr_daily_volume_tasks import (
    CXRDailyForecastTask,
    CXRDailyForecastUnder1YearTask,
    CXRDailyForecastUnder6MonthsTask,
)


TASK_REGISTRY = {
    "CXRDailyForecastTask": CXRDailyForecastTask,
    "CXRDailyForecastUnder6MonthsTask": CXRDailyForecastUnder6MonthsTask,
    "CXRDailyForecastUnder1YearTask": CXRDailyForecastUnder1YearTask,
}


def experiment_last_value(use_context=False):
    return LastValueBaseline(use_context=use_context)


def experiment_direct_context_prompt(
    model=None,
    use_context=True,
    temperature=0.2,
    timeout_seconds=120,
    dry_run=False,
):
    return DirectContextPromptBaseline(
        model=model,
        use_context=use_context,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
    )


def available_experiments():
    return {
        name.removeprefix("experiment_"): func
        for name, func in globals().items()
        if name.startswith("experiment_") and inspect.isfunction(func)
    }


def main():
    parser = argparse.ArgumentParser(description="Run custom CXR baseline experiments.")
    parser.add_argument("--exp-spec", help="Path to a JSON experiment spec.")
    parser.add_argument(
        "--output",
        default="./benchmark_results",
        help="Output directory for experiment artifacts.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of forecast samples to generate per seed.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds to evaluate per experiment.",
    )
    parser.add_argument(
        "--list-exps",
        action="store_true",
        help="List available experiment methods and exit.",
    )
    parser.add_argument(
        "--context-text",
        help="Optional natural-language context override applied to every experiment task_config.",
    )
    args = parser.parse_args()

    experiments = available_experiments()
    if args.list_exps:
        for name in sorted(experiments):
            print(name)
        return 0
    if not args.exp_spec:
        parser.error("--exp-spec is required unless --list-exps is used.")

    with open(args.exp_spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    for exp in spec:
        label = exp["label"]
        method_name = exp["method"]
        task_name = exp["task_class"]
        task_config = dict(exp.get("task_config", {}))
        method_kwargs = exp.get("method_kwargs", {})

        if args.context_text:
            task_config["context_text"] = args.context_text

        if method_name not in experiments:
            raise RuntimeError(f"Unknown experiment method: {method_name}")
        if task_name not in TASK_REGISTRY:
            raise RuntimeError(f"Unknown task class: {task_name}")

        baseline = experiments[method_name](**method_kwargs)
        task_cls = TASK_REGISTRY[task_name]
        exp_output = output_root / label
        exp_output.mkdir(parents=True, exist_ok=True)

        rows = []
        for seed in range(1, args.seeds + 1):
            rows.append(
                evaluate_task_instance(
                    task_cls=task_cls,
                    seed=seed,
                    method_callable=baseline,
                    n_samples=args.n_samples,
                    task_config=task_config,
                    output_folder=exp_output,
                )
            )

        raw_df, summary_df = summarize_results(rows)
        raw_df.to_csv(exp_output / "seed_results.csv", index=False)
        summary_df.to_csv(exp_output / "summary_results.csv", index=False)

        all_summary_rows.append(
            {
                "label": label,
                "task": task_name,
                "method": method_name,
                "use_context": method_kwargs.get("use_context", False),
                "summary_path": str(exp_output / "summary_results.csv"),
                "seed_results_path": str(exp_output / "seed_results.csv"),
            }
        )

    import pandas as pd

    pd.DataFrame(all_summary_rows).to_csv(output_root / "experiment_index.csv", index=False)
    print(f"Wrote experiment outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
