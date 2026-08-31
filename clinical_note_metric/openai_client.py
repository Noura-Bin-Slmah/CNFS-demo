"""OpenAI-backed implementation of the CNFS LLM client protocol."""

from __future__ import annotations

import json
from typing import Any


class OpenAIJudgeClient:
    """LLM client that calls the OpenAI Responses API.

    The OpenAI SDK reads the API key from the ``OPENAI_API_KEY`` environment
    variable by default.
    """

    def __init__(self, *, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is required. Install it with: pip install openai"
            ) from exc

        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None,
        temperature: float,
    ) -> str | dict[str, Any]:
        """Generate a JSON response from OpenAI."""

        resolved_model = model or "gpt-4.1-mini-2025-04-14"
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "instructions": system_prompt,
            "input": user_prompt,
            # Stable per model: hints the API to route repeat calls with the
            # same static instructions prefix to a cache-warm backend.
            "prompt_cache_key": f"cnfs-judge-{resolved_model}",
            "text": {"format": {"type": "json_object"}},
        }
        # Reasoning-family models (o1/o3/o4/gpt-5+) only support the
        # default temperature and reject an explicit value.
        if not resolved_model.startswith(("o1", "o3", "o4", "gpt-5")):
            kwargs["temperature"] = temperature

        response = self.client.responses.create(**kwargs)
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        # Defensive handling for SDK/object shape changes.
        response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return json.dumps(response_dict)
