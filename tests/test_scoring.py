from clinical_note_metric.config import MetricConfig
from clinical_note_metric.models import ClinicalFact, FactMatch, MatchClassification
from clinical_note_metric.scoring import CNFSScorer


def fact(fact_id: str, concept: str, section: str = "Assessment") -> ClinicalFact:
    return ClinicalFact(
        id=fact_id,
        section=section,
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


def test_sections_are_weighted_equally_not_by_fact_count():
    # A large, perfect Assessment section should not dilute a small but
    # completely missed Plan section (or vice versa) — each section
    # contributes equally to the overall score, regardless of how many
    # facts it contains.
    assessment_facts = [fact(f"gt_a{i}", f"assessment finding {i}", section="Assessment") for i in range(10)]
    plan_fact = fact("gt_p1", "plan action", section="Plan")

    matches = [
        match(af, MatchClassification.CORRECT, generated_fact=fact(f"gen_a{i}", f"assessment finding {i}"))
        for i, af in enumerate(assessment_facts)
    ] + [match(plan_fact, MatchClassification.MISSING)]

    scores = CNFSScorer(MetricConfig()).dimension_scores(matches=matches, extras=[], generated_fact_count=10)

    # Pooling all 11 facts together would give (10*1.0)/11*100 ≈ 90.9.
    # Equal per-section weighting averages Assessment's 100 with Plan's 0.
    assert scores.completeness == 50.0
