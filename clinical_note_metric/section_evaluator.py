"""Section-level score aggregation."""

from __future__ import annotations

from clinical_note_metric.models import (
    ExtraGeneratedFact,
    GeneratedFactClassification,
    MatchClassification,
    SECTIONS,
    SectionScore,
)
from clinical_note_metric.scoring import CAPTURE_SCORES
from clinical_note_metric.utils import clamp_score


class SectionEvaluator:
    """Computes per-section metrics."""

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
                correctness = completeness
            else:
                completeness = correctness = 100.0

            unsupported_count = sum(
                1
                for extra in extras
                if extra.generated_fact.section == section
                and extra.classification == GeneratedFactClassification.UNSUPPORTED
            )
            result[section] = SectionScore(
                completeness=clamp_score(completeness),
                correctness=clamp_score(correctness),
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
