from clinical_note_metric.config import MetricConfig
from clinical_note_metric.models import ClinicalFact, FactMatch, MatchClassification
from clinical_note_metric.scoring import CNFSScorer


def fact(fact_id: str, concept: str) -> ClinicalFact:
    return ClinicalFact(
        id=fact_id,
        section="Assessment",
        concept=concept,
        evidence_text=concept,
    )


def match(
    ground_truth_fact: ClinicalFact,
    classification: MatchClassification,
    *,
    generated_fact: ClinicalFact | None = None,
) -> FactMatch:
    return FactMatch(
        ground_truth_fact=ground_truth_fact,
        generated_fact=generated_fact,
        classification=classification,
        section_score=1.0 if generated_fact else 0.0,
        reason="test match",
    )


def test_correctness_excludes_missing_facts():
    gt_a = fact("gt_0001", "documented problem")
    gt_b = fact("gt_0002", "missing problem")
    gen_a = fact("gen_0001", "documented problem")

    scores = CNFSScorer(MetricConfig()).dimension_scores(
        matches=[
            match(gt_a, MatchClassification.CORRECT, generated_fact=gen_a),
            match(gt_b, MatchClassification.MISSING),
        ],
        extras=[],
        generated_fact_count=1,
    )

    assert scores.completeness == 50.0
    assert scores.correctness == 100.0
