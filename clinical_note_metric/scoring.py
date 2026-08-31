"""Deterministic CNFS score calculations."""

from __future__ import annotations

from clinical_note_metric.config import MetricConfig
from clinical_note_metric.models import (
    SECTIONS,
    ExtraGeneratedFact,
    FactMatch,
    GeneratedFactClassification,
    MatchClassification,
    ScoreBreakdown,
)
from clinical_note_metric.utils import clamp_score, safe_mean


CAPTURE_SCORES = {
    MatchClassification.CORRECT: 1.0,
    MatchClassification.PARTIAL: 0.5,
    MatchClassification.INCORRECT: 0.0,
    MatchClassification.CONTRADICTION: 0.0,
    MatchClassification.MISSING: 0.0,
}


def weighted_overall(scores: ScoreBreakdown, weights: dict[str, float]) -> float:
    """Combine the three dimensions into one score using the configured
    weights — shared by the note-level overall and each section's own
    overall, so both use identical math."""

    total_weight = sum(weights.values())
    weighted = (
        scores.completeness * weights["completeness"]
        + scores.correctness * weights["correctness"]
        + scores.supported_content * weights["supported_content"]
    )
    return clamp_score(weighted / total_weight)


class CNFSScorer:
    """Calculates dimension and final scores."""

    def __init__(self, config: MetricConfig):
        self.config = config

    def dimension_scores(
        self,
        matches: list[FactMatch],
        extras: list[ExtraGeneratedFact],
        generated_fact_count: int,
    ) -> ScoreBreakdown:
        """Score each section independently, then average across sections that
        had applicable facts — a section with nothing to grade does not
        contribute to (or dilute) the overall score."""

        completeness_scores: list[float] = []
        correctness_scores: list[float] = []
        supported_scores: list[float] = []

        for section in SECTIONS:
            section_matches = [m for m in matches if m.ground_truth_fact.section == section]
            if section_matches:
                completeness_scores.append(self._reference_fact_score(section_matches))

            attempted = [m for m in section_matches if m.classification != MatchClassification.MISSING]
            if attempted:
                correctness_scores.append(self._attempted_fact_score(attempted))

            section_generated_count = sum(
                1 for m in matches if m.generated_fact and m.generated_fact.section == section
            ) + sum(1 for extra in extras if extra.generated_fact.section == section)
            if section_generated_count:
                section_unsupported = sum(
                    1
                    for extra in extras
                    if extra.generated_fact.section == section
                    and extra.classification == GeneratedFactClassification.UNSUPPORTED
                )
                supported_scores.append((1 - section_unsupported / section_generated_count) * 100)

        return ScoreBreakdown(
            completeness=clamp_score(safe_mean(completeness_scores)),
            correctness=clamp_score(safe_mean(correctness_scores)),
            supported_content=clamp_score(safe_mean(supported_scores)),
        )

    def base_score(self, scores: ScoreBreakdown) -> float:
        return weighted_overall(scores, self.config.weights)

    @staticmethod
    def _reference_fact_score(matches: list[FactMatch]) -> float:
        points = sum(CAPTURE_SCORES[MatchClassification(match.classification)] for match in matches)
        return (points / len(matches)) * 100

    @staticmethod
    def _attempted_fact_score(attempted: list[FactMatch]) -> float:
        points = sum(CAPTURE_SCORES[MatchClassification(match.classification)] for match in attempted)
        return (points / len(attempted)) * 100
