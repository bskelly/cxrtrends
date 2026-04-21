"""Minimal baselines for custom CXR experiment wiring."""

from __future__ import annotations

import json
import numpy as np
import os
import re
from urllib import error, request


class LastValueBaseline:
    """Repeat the final observed count across the forecast horizon.

    The context-enabled variant uses the same deterministic forecast for now,
    but it captures the natural-language scenario text in `extra_info` so the
    wiring is ready for later replacement with an actual context-aware model.
    """

    __version__ = "0.0.1"

    def __init__(self, use_context: bool = False):
        self.use_context = use_context

    def __call__(self, task_instance, n_samples):
        last_value = float(task_instance.past_time["exam_count"].iloc[-1])
        horizon = len(task_instance.future_time)
        samples = np.full((n_samples, horizon, 1), fill_value=last_value, dtype=float)
        extra_info = {
            "baseline": "last_value",
            "use_context": self.use_context,
            "scenario_text": task_instance.scenario if self.use_context else None,
            "series_id": getattr(task_instance, "series_id", None),
        }
        return samples, extra_info

    @property
    def cache_name(self):
        return f"{self.__class__.__name__}_use_context={self.use_context}"


class DirectContextPromptBaseline:
    """OpenAI-compatible prompt baseline where natural-language context affects forecasts.

    This is the real Context-Is-Key style integration point for the custom task
    family: the model sees the history and, when enabled, the natural-language
    scenario text and can update the forecast accordingly.
    """

    __version__ = "0.0.1"

    def __init__(
        self,
        model: str | None = None,
        use_context: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout_seconds: int = 120,
        dry_run: bool = False,
    ):
        self.model = model or os.environ.get("CIK_OPENAI_MODEL", "gpt-4o-mini")
        self.use_context = use_context
        self.api_key = api_key or os.environ.get("CIK_OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("CIK_OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.dry_run = dry_run

        if not self.dry_run and not self.api_key:
            raise RuntimeError(
                "CIK_OPENAI_API_KEY is required for the direct context prompt baseline."
            )

    def make_prompt(self, task_instance) -> str:
        history = "\n".join(
            f"({idx.strftime('%Y-%m-%d')}, {int(value) if float(value).is_integer() else float(value)})"
            for idx, value in task_instance.past_time["exam_count"].items()
        )
        future_dates = [
            idx.strftime("%Y-%m-%d") for idx in task_instance.future_time.index
        ]

        context_lines = []
        if self.use_context:
            if task_instance.background:
                context_lines.append(f"Background: {task_instance.background}")
            if task_instance.scenario:
                context_lines.append(f"Scenario: {task_instance.scenario}")
        context_block = "\n".join(context_lines) if context_lines else "No additional context."

        return f"""
You are a forecasting assistant. Forecast future daily exam counts from the provided history.

Use the natural-language context when it is provided. If the scenario says something like
"a new vaccine was implemented on [date] to reduce respiratory diseases", then your forecast
should reflect the expected change after that date if it falls inside or influences the forecast window.

Return only valid JSON in this exact format:
{{
  "forecast": [
    {{"date": "YYYY-MM-DD", "value": 0}}
  ]
}}

Task context:
{context_block}

Historical daily counts:
{history}

Forecast these dates:
{future_dates}
""".strip()

    def _chat_completion(self, prompt: str, n_samples: int) -> dict:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "n": n_samples,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise forecasting assistant that returns JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat completion request failed: {exc.code} {body}") from exc

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise RuntimeError("Model output did not contain JSON.")
            return json.loads(match.group(0))

    def _parse_choice(self, content: str, future_index) -> list[float]:
        payload = self._extract_json(content)
        forecast = payload.get("forecast")
        if not isinstance(forecast, list):
            raise RuntimeError("Model output JSON must contain a 'forecast' list.")

        value_by_date = {}
        for row in forecast:
            if not isinstance(row, dict):
                raise RuntimeError("Each forecast row must be a JSON object.")
            if "date" not in row or "value" not in row:
                raise RuntimeError("Each forecast row must contain 'date' and 'value'.")
            value_by_date[str(row["date"])] = float(row["value"])

        return [value_by_date[idx.strftime("%Y-%m-%d")] for idx in future_index]

    def __call__(self, task_instance, n_samples):
        prompt = self.make_prompt(task_instance)
        horizon = len(task_instance.future_time)

        if self.dry_run:
            samples = np.full((n_samples, horizon, 1), np.nan, dtype=float)
            return samples, {
                "baseline": "direct_context_prompt",
                "use_context": self.use_context,
                "prompt": prompt,
                "dry_run": True,
                "series_id": getattr(task_instance, "series_id", None),
            }

        response = self._chat_completion(prompt=prompt, n_samples=n_samples)
        choices = response.get("choices", [])
        parsed = []
        raw_outputs = []
        for choice in choices:
            content = choice.get("message", {}).get("content", "")
            raw_outputs.append(content)
            parsed.append(
                self._parse_choice(content, future_index=task_instance.future_time.index)
            )

        if len(parsed) != n_samples:
            raise RuntimeError(
                f"Expected {n_samples} parsed forecasts, received {len(parsed)}."
            )

        samples = np.array(parsed, dtype=float)[:, :, None]
        usage = response.get("usage", {})
        return samples, {
            "baseline": "direct_context_prompt",
            "use_context": self.use_context,
            "series_id": getattr(task_instance, "series_id", None),
            "scenario_text": task_instance.scenario if self.use_context else None,
            "prompt": prompt,
            "raw_outputs": raw_outputs,
            "usage": usage,
            "model": self.model,
        }

    @property
    def cache_name(self):
        return (
            f"{self.__class__.__name__}_model={self.model}_use_context={self.use_context}"
        )
