"""Deterministic CNFS score calculations."""

from __future__ import annotations

from clinical_note_metric.config import MetricConfig
from clinical_note_metric.models import (
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
        completeness = self._reference_fact_score(matches)
        correctness = self._attempted_fact_score(matches)
        unsupported = sum(
            1 for extra in extras if extra.classification == GeneratedFactClassification.UNSUPPORTED
        )
        unsupported_rate = unsupported / generated_fact_count if generated_fact_count else 0.0
        section = safe_mean(
            [
                match.section_score * 100
                for match in matches
                if match.classification != MatchClassification.MISSING
            ]
        )
        return ScoreBreakdown(
            completeness=clamp_score(completeness),
            correctness=clamp_score(correctness),
            supported_content=clamp_score((1 - unsupported_rate) * 100),
            section_placement=clamp_score(section),
        )

    def base_score(self, scores: ScoreBreakdown) -> float:
        weights = self.config.weights
        total_weight = sum(weights.values())
        weighted = (
            scores.completeness * weights["completeness"]
            + scores.correctness * weights["correctness"]
            + scores.supported_content * weights["supported_content"]
            + scores.section_placement * weights["section_placement"]
        )
        return clamp_score(weighted / total_weight)

    @staticmethod
    def _reference_fact_score(matches: list[FactMatch]) -> float:
        if not matches:
            return 100.0
        points = sum(CAPTURE_SCORES[MatchClassification(match.classification)] for match in matches)
        return (points / len(matches)) * 100

    @staticmethod
    def _attempted_fact_score(matches: list[FactMatch]) -> float:
        attempted = [
            match
            for match in matches
            if match.classification != MatchClassification.MISSING
        ]
        if not attempted:
            return 100.0
        points = sum(CAPTURE_SCORES[MatchClassification(match.classification)] for match in attempted)
        return (points / len(attempted)) * 100
