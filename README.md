# Context Is Key Forecasting Adaptation

## Phase 1: Deterministic Data Ingestion

This workspace contains the first migration phase for adapting the
`context-is-key-forecasting` repository to the CXR trends use case.

### Expected input schema

The ingestion script expects an Excel workbook with at least the following
columns on the selected sheet:

- `Facility`
- `Accession Number`
- `Exam Description`
- `Imaged DTTM`
- `Age`
- `Age Category`
- `Patient Class`
- `Order Location`
- `Ref Office Code`
- `Admission DTTM`
- `Discharge DTTM`
- `Exam Reason`
- `Location Type`
- `Attending Specialty`
- `PACS Work Group`
- `Imaged Day of Week`
- `Imaged Hour of Day`
- `9-5 Flag`
- `Prelim Texts`

### Produced normalized schema

The converter writes a deterministic normalized table with one row per valid
exam event and the following columns:

- `source_workbook`
- `source_sheet`
- `exam_id`
- `facility_id`
- `exam_description`
- `imaged_ts`
- `imaged_date`
- `imaged_day_of_week`
- `imaged_hour_of_day`
- `is_business_hours_derived`
- `admission_ts`
- `discharge_ts`
- `age_years`
- `age_category`
- `patient_class`
- `order_location`
- `ref_office_code`
- `location_type`
- `attending_specialty`
- `pacs_work_group`
- `exam_reason_text`
- `prelim_text`
- `imaged_day_of_week_source`
- `imaged_hour_of_day_source`
- `business_hours_flag_source`

Rows missing the core exam identifier or imaging timestamp are excluded from the
normalized output and counted in a metadata report. Duplicate valid exam IDs are
treated as a hard validation error.

### CLI usage

```bash
python scripts/normalize_cxr_trends.py \
  --input "/path/to/cxrtrends.xlsx" \
  --output data/normalized/cxr_trends.parquet \
  --format parquet
```

## Phase 2: Daily Forecast Task Integration

The Phase 2 task adapter consumes the normalized dataset from Phase 1 and maps
it into the repository-style `past_time` / `future_time` forecasting interface.

### Added task interfaces

- `cik_benchmark.data.cxr_daily.CXRDailyVolumeAdapter`
- `cik_benchmark.tasks.cxr_daily_volume_tasks.CXRDailyForecastTask`
- `cik_benchmark.tasks.cxr_daily_volume_tasks.CXRDailyForecastUnder6MonthsTask`
- `cik_benchmark.tasks.cxr_daily_volume_tasks.CXRDailyForecastUnder1YearTask`

### Supported task configuration

- `normalized_path`: path to the normalized CSV or parquet file
- `frequency`: currently validated as daily (`D`)
- `horizon`: number of forecast periods
- `history_length`: number of past periods to expose
- `age_upper_years`: optional age threshold for the cohort
- `series_mode`: `single` or `panel`
- `panel_column`: required for panel mode
- `entity_id`: optional fixed panel entity
- `context_text`: optional free-text scenario placeholder for future clinical context
- `context_builder`: optional callable hook for future generated context

### Smoke test

```bash
python scripts/smoke_test_cxr_task.py
```

## Phase 3: Experiment Wiring

The repository now includes a minimal experiment runner for the custom CXR task
family:

- `run_baselines.py`
- `cik_benchmark/baselines/cxr_baselines.py`
- `cik_benchmark/evaluation.py`
- `experiments/cxr_daily_no_context.json`
- `experiments/cxr_daily_with_context.json`
- `experiments/cxr_daily_compare.json`

### Supported runs

1. No-context baseline:
   history-only last-value forecast
2. Context-enabled baseline:
   same deterministic baseline for now, but with the natural-language clinical
   context plumbed through `task.scenario` and preserved in experiment artifacts

This keeps the wiring compatible with the original Context is Key idea: the text
context exists in the task layer already, so you can later replace the current
baseline with an actual context-aware LLM or prompt-based forecaster without
changing the task or spec shape.

### Example commands

```bash
/Users/brendankelly/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
run_baselines.py \
  --exp-spec experiments/cxr_daily_compare.json \
  --output benchmark_results/cxr_compare \
  --n-samples 5 \
  --seeds 2
```

## Natural-Language Context As A Real Model Input

The repository now also supports a prompt-based context forecaster:

- method: `direct_context_prompt`
- baseline file: `cik_benchmark/baselines/cxr_baselines.py`

This method sends the history plus natural-language task context to an
OpenAI-compatible chat completion model and asks it to forecast the requested
future dates. The context string is not a placeholder in this path: it is part
of the actual model prompt and can change the returned forecast.

Example scenario text:

`A new vaccine was implemented on 2024-10-01 to reduce respiratory diseases.`

You can pass context either:

- in the experiment spec via `task_config.context_text`
- or at run time with `--context-text`

Required environment variable for the prompt model:

- `CIK_OPENAI_API_KEY`

Optional:

- `CIK_OPENAI_MODEL`
- `CIK_OPENAI_BASE_URL`

Example run with a direct prompt model and explicit natural-language context:

```bash
CIK_OPENAI_API_KEY=your_key_here \
/Users/brendankelly/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
run_baselines.py \
  --exp-spec experiments/cxr_daily_direct_prompt_with_context.json \
  --output benchmark_results/cxr_direct_prompt_with_context \
  --n-samples 2 \
  --seeds 1 \
  --context-text "A new vaccine was implemented on 2024-10-01 to reduce respiratory diseases."
```

## Phase 4: Metrics And Reporting

The custom task family now writes simple tabular outputs and a compact report.

### Metrics

The current custom metrics are point-forecast metrics computed from the median
forecast sample:

- `mae`
- `rmse`
- `bias`
- `wape`
- `smape`

These are implemented in:

- `cik_benchmark/metrics/point_forecast.py`

### Per-seed tabular outputs

Each evaluated seed now writes:

- `past_time.csv`
- `future_time.csv`
- `samples.csv`
- `forecast_table.csv`
- `evaluation.json`
- `extra_info.json`
- `context.txt`

`forecast_table.csv` is the quickest artifact for inspecting predictions
against actuals period by period.

### Compact report script

```bash
/Users/brendankelly/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
scripts/report_cxr_results.py \
  --results-root benchmark_results/cxr_compare
```

This produces:

- `report_seed_results.csv`
- `report_summary.csv`
- `report.md`
