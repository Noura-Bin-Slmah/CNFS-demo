"""Section-level score aggregation."""

from __future__ import annotations

from clinical_note_metric.config import MetricConfig
from clinical_note_metric.models import (
    ExtraGeneratedFact,
    GeneratedFactClassification,
    MatchClassification,
    SECTIONS,
    ScoreBreakdown,
    SectionScore,
)
from clinical_note_metric.scoring import CAPTURE_SCORES, weighted_overall
from clinical_note_metric.utils import clamp_score


class SectionEvaluator:
    """Computes per-section metrics, including each section's own overall
    score using the same weights and formula as the note-level overall."""

    def __init__(self, config: MetricConfig):
        self.config = config

    def evaluate(self, matches, extras: list[ExtraGeneratedFact]) -> dict[str, SectionScore]:
        result = {section: SectionScore() for section in SECTIONS}
        for section in SECTIONS:
            section_matches = [m for m in matches if m.ground_truth_fact.section == section]
            if section_matches:
                completeness = (
                    sum(CAPTURE_SCORES[MatchClassification(m.classification)] for m in section_matches)
                    / len(section_matches)
                    * 100
                )
            else:
                completeness = 100.0

            attempted = [m for m in section_matches if m.classification != MatchClassification.MISSING]
            if attempted:
                correctness = (
                    sum(CAPTURE_SCORES[MatchClassification(m.classification)] for m in attempted)
                    / len(attempted)
                    * 100
                )
            else:
                correctness = 100.0

            section_generated_count = sum(
                1 for m in matches if m.generated_fact and m.generated_fact.section == section
            ) + sum(1 for extra in extras if extra.generated_fact.section == section)
            unsupported_count = sum(
                1
                for extra in extras
                if extra.generated_fact.section == section
                and extra.classification == GeneratedFactClassification.UNSUPPORTED
            )
            if section_generated_count:
                supported_content = (1 - unsupported_count / section_generated_count) * 100
            else:
                supported_content = 100.0

            completeness = clamp_score(completeness)
            correctness = clamp_score(correctness)
            supported_content = clamp_score(supported_content)
            overall = weighted_overall(
                ScoreBreakdown(
                    completeness=completeness,
                    correctness=correctness,
                    supported_content=supported_content,
                ),
                self.config.weights,
            )

            result[section] = SectionScore(
                fact_count=len(section_matches),
                generated_fact_count=section_generated_count,
                completeness=completeness,
                correctness=correctness,
                supported_content=supported_content,
                overall=overall,
                correct_fact_count=sum(
                    1 for m in section_matches if m.classification == MatchClassification.CORRECT
                ),
                partial_fact_count=sum(
                    1 for m in section_matches if m.classification == MatchClassification.PARTIAL
                ),
                unsupported_fact_count=unsupported_count,
                missing_fact_count=sum(
                    1 for m in section_matches if m.classification == MatchClassification.MISSING
                ),
                incorrect_fact_count=sum(
                    1 for m in section_matches if m.classification == MatchClassification.INCORRECT
                ),
                contradiction_count=sum(
                    1 for m in section_matches if m.classification == MatchClassification.CONTRADICTION
                ),
            )
        return result
