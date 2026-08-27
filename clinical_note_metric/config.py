"""Configuration for CNFS."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


DEFAULT_WEIGHTS = {
    "completeness": 0.30,
    "correctness": 0.40,
    "supported_content": 0.20,
    "section_placement": 0.10,
}


class MetricConfig(BaseModel):
    """Runtime configuration for the metric."""

    metric_version: str = "1.0"
    prompt_version: str = "1.0"
    model: str | None = None
    temperature: float = 0.0
    weights: dict[str, float] = Field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    llm_retry_attempts: int = 2
    use_llm_extraction: bool = True
    use_llm_matching: bool = True
    lexical_normalization: dict[str, str] = Field(default_factory=dict)
    semantic_match_threshold: float = 0.72
    acceptable_section_pairs: set[tuple[str, str]] = Field(
        default_factory=lambda: {
            ("Problem List", "Assessment"),
            ("Assessment", "Problem List"),
            ("Assessment", "Plan"),
            ("Plan", "Assessment"),
        }
    )

    @field_validator("weights")
    @classmethod
    def weights_must_be_complete(cls, value: dict[str, float]) -> dict[str, float]:
        missing = set(DEFAULT_WEIGHTS) - set(value)
        if missing:
            raise ValueError(f"Missing weights: {sorted(missing)}")
        total = sum(value.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        return value
