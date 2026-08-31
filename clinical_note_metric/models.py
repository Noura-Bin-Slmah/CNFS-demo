"""Structured models for the Clinical Note Fidelity Score."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SECTIONS = ("Problem List", "Subjective", "Objective", "Assessment", "Plan")


class FactStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    PLANNED = "planned"
    COMPLETED = "completed"
    CURRENT = "current"
    PREVIOUS = "previous"
    UNKNOWN = "unknown"


class MatchClassification(str, Enum):
    CORRECT = "CORRECT"
    PARTIAL = "PARTIAL"
    INCORRECT = "INCORRECT"
    CONTRADICTION = "CONTRADICTION"
    MISSING = "MISSING"


class GeneratedFactClassification(str, Enum):
    SUPPORTED_BUT_ABSENT_FROM_GT = "SUPPORTED_BUT_ABSENT_FROM_GT"
    UNSUPPORTED = "UNSUPPORTED"


class ClinicalErrorEventType(str, Enum):
    MISSING_DETAIL = "MISSING_DETAIL"
    NEGATION_ERROR = "NEGATION_ERROR"
    LATERALITY_ERROR = "LATERALITY_ERROR"
    MEDICATION_DETAIL_OMISSION = "MEDICATION_DETAIL_OMISSION"
    MEDICATION_ERROR = "MEDICATION_ERROR"
    DOSE_ERROR = "DOSE_ERROR"
    ROUTE_ERROR = "ROUTE_ERROR"
    ALLERGY_ERROR = "ALLERGY_ERROR"
    UNSUPPORTED_MEDICATION = "UNSUPPORTED_MEDICATION"
    UNSUPPORTED_DIAGNOSIS = "UNSUPPORTED_DIAGNOSIS"
    UNSUPPORTED_OBJECTIVE_FINDING = "UNSUPPORTED_OBJECTIVE_FINDING"
    CRITICAL_OMISSION = "CRITICAL_OMISSION"
    SAFETY_CRITICAL_CONTRADICTION = "SAFETY_CRITICAL_CONTRADICTION"
    SOURCE_CERTAINTY_TRANSFORMATION = "SOURCE_CERTAINTY_TRANSFORMATION"


class ClinicalFact(BaseModel):
    """An atomic clinical assertion extracted from a note."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    section: Literal["Problem List", "Subjective", "Objective", "Assessment", "Plan"]
    concept: str
    value: str | None = None
    status: FactStatus = FactStatus.UNKNOWN
    negation: bool | None = None
    temporality: str | None = None
    laterality: str | None = None
    anatomical_location: str | None = None
    severity: str | None = None
    duration: str | None = None
    frequency: str | None = None
    dose: str | None = None
    route: str | None = None
    medication_name: str | None = None
    source_or_speaker: str | None = None
    certainty: str | None = None
    evidence_text: str
    is_safety_relevant: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("section", mode="before")
    @classmethod
    def normalize_section(cls, value: Any) -> Any:
        """Accept common LLM section variants."""

        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("_", " ")
        section_map = {
            "problem list": "Problem List",
            "problems": "Problem List",
            "subjective": "Subjective",
            "objective": "Objective",
            "assessment": "Assessment",
            "plan": "Plan",
        }
        return section_map.get(normalized, value)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: Any) -> Any:
        """Coerce common judge wording into the supported status enum."""

        if value is None or value == "":
            return FactStatus.UNKNOWN
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("_", " ")
        status_map = {
            "active": FactStatus.PRESENT,
            "present": FactStatus.PRESENT,
            "positive": FactStatus.PRESENT,
            "positive finding": FactStatus.PRESENT,
            "noted": FactStatus.PRESENT,
            "ongoing": FactStatus.CURRENT,
            "current": FactStatus.CURRENT,
            "controlled": FactStatus.CURRENT,
            "uncontrolled": FactStatus.CURRENT,
            "absent": FactStatus.ABSENT,
            "negative": FactStatus.ABSENT,
            "negated": FactStatus.ABSENT,
            "denied": FactStatus.ABSENT,
            "not present": FactStatus.ABSENT,
            "normal": FactStatus.NORMAL,
            "within normal limits": FactStatus.NORMAL,
            "wnl": FactStatus.NORMAL,
            "abnormal": FactStatus.ABNORMAL,
            "planned": FactStatus.PLANNED,
            "plan": FactStatus.PLANNED,
            "ordered": FactStatus.PLANNED,
            "recommended": FactStatus.PLANNED,
            "completed": FactStatus.COMPLETED,
            "done": FactStatus.COMPLETED,
            "performed": FactStatus.COMPLETED,
            "previous": FactStatus.PREVIOUS,
            "prior": FactStatus.PREVIOUS,
            "historical": FactStatus.PREVIOUS,
            "unknown": FactStatus.UNKNOWN,
            "unspecified": FactStatus.UNKNOWN,
        }
        return status_map.get(normalized, value)


class FactMatch(BaseModel):
    """Classification of one ground-truth fact against a generated fact in the same section."""

    model_config = ConfigDict(use_enum_values=True)

    ground_truth_fact: ClinicalFact
    generated_fact: ClinicalFact | None = None
    classification: MatchClassification
    reason: str


class ExtraGeneratedFact(BaseModel):
    """Generated fact absent from the ground truth."""

    model_config = ConfigDict(use_enum_values=True)

    generated_fact: ClinicalFact
    classification: GeneratedFactClassification
    reason: str


class SectionPlacementIssue(BaseModel):
    """A ground-truth fact marked MISSING in its own section whose content was
    actually found in a different section of the generated note. Informational
    only — does not change any classification or score, the same way a
    clinical error event is reported separately from the fidelity score."""

    model_config = ConfigDict(use_enum_values=True)

    ground_truth_fact: ClinicalFact
    generated_fact: ClinicalFact
    reason: str


class ClinicalErrorEvent(BaseModel):
    """A clinical documentation fidelity error event."""

    model_config = ConfigDict(use_enum_values=True)

    type: ClinicalErrorEventType
    severity: Literal["NONE", "LOW", "MODERATE", "HIGH", "CRITICAL"] = "LOW"
    error_group_id: str
    primary_error: ClinicalErrorEventType | None = None
    secondary_labels: list[str] = Field(default_factory=list)
    is_safety_relevant: bool = True
    safety_reason: str = ""
    clinical_consequence: str = ""
    evidence_text: str = ""
    ground_truth_fact: ClinicalFact | None = None
    generated_fact: ClinicalFact | None = None
    reason: str


class ScoreBreakdown(BaseModel):
    completeness: float
    correctness: float
    supported_content: float


class CountBreakdown(BaseModel):
    ground_truth_facts: int
    generated_facts: int
    correct: int
    partial: int
    incorrect: int
    contradictions: int
    missing: int
    unsupported: int
    supported_but_absent_from_gt: int
    clinical_error_events: int
    section_placement_issues: int


class SectionScore(BaseModel):
    fact_count: int = 0
    generated_fact_count: int = 0
    completeness: float = 100.0
    correctness: float = 100.0
    supported_content: float = 100.0
    overall: float = 100.0
    correct_fact_count: int = 0
    partial_fact_count: int = 0
    unsupported_fact_count: int = 0
    missing_fact_count: int = 0
    incorrect_fact_count: int = 0
    contradiction_count: int = 0


class CNFSResult(BaseModel):
    """Serializable output for one CNFS evaluation."""

    metric_name: str = "Clinical Note Fidelity Score"
    metric_version: str
    final_score: float
    overall_fidelity_score: float
    fidelity_score: float
    scores: ScoreBreakdown
    counts: CountBreakdown
    section_scores: dict[str, SectionScore]
    fact_matches: list[FactMatch]
    unsupported_facts: list[ExtraGeneratedFact]
    clinical_error_events: list[ClinicalErrorEvent]
    section_placement_issues: list[SectionPlacementIssue] = Field(default_factory=list)
    transcript_used: bool
    limitations: list[str] = Field(default_factory=list)
    summary: str
    raw_judge_reasoning_summary: list[str] = Field(default_factory=list)
