"""Provider-neutral JSON LLM judge interface."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from pydantic import ValidationError

from clinical_note_metric.config import MetricConfig
from clinical_note_metric.prompts import SYSTEM_GUARDRAILS

LOGGER = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol implemented by OpenAI, LiteLLM, or another provider wrapper."""

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None,
        temperature: float,
    ) -> str | dict[str, Any]:
        """Return a JSON string or already-decoded JSON object."""


class JsonLLMJudge:
    """Small retrying wrapper that validates JSON shape at the boundary."""

    def __init__(self, client: LLMClient, config: MetricConfig):
        self.client = client
        self.config = config

    def ask_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.llm_retry_attempts + 1):
            try:
                response = self.client.generate_json(
                    system_prompt=SYSTEM_GUARDRAILS,
                    user_prompt=prompt,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
                if isinstance(response, dict):
                    return response
                return json.loads(response)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                last_error = exc
                LOGGER.warning("Malformed LLM JSON on attempt %s: %s", attempt + 1, exc)
        raise ValueError("LLM judge returned malformed JSON.") from last_error
