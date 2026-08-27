import pytest

from clinical_note_metric import ClinicalNoteEvaluator, MetricConfig
from clinical_note_metric.models import ClinicalFact


class SingleCallFakeLLMClient:
    def __init__(self, *, omit_extra_once: bool = False):
        self.calls = 0
        self.omit_extra_once = omit_extra_once

    def generate_json(self, *, system_prompt, user_prompt, model, temperature):
        self.calls += 1
        if "Evaluate a generated medical note" not in user_prompt:
            raise AssertionError("Evaluator should use the single-call prompt only.")
        payload = self._payload(user_prompt)
        if self.omit_extra_once and self.calls == 1 and payload["unsupported_facts"]:
            payload["unsupported_facts"] = payload["unsupported_facts"][:-1]
        return payload

    def _payload(self, prompt):
        gt_note = self._between(prompt, "Ground-truth note:", "Generated note:")
        gen_note = self._between(prompt, "Generated note:", "Transcript, if provided:")
        transcript = prompt.split("Transcript, if provided:", 1)[1]
        gt_facts = self._facts(gt_note, "gt")
        gen_facts = self._facts(gen_note, "gen")
        transcript_facts = self._facts(transcript, "transcript")
        used = set()
        matches = []

        for gt_fact in gt_facts:
            candidate = next(
                (fact for fact in gen_facts if fact["concept"] == gt_fact["concept"] and fact["id"] not in used),
                None,
            )
            if candidate is None:
                matches.append(
                    {
                        "ground_truth_fact_id": gt_fact["id"],
                        "generated_fact_id": None,
                        "classification": "MISSING",
                        "section_score": 0,
                        "detail_score": 0,
                        "reason": "No generated fact matched.",
                    }
                )
                continue
            used.add(candidate["id"])
            classification, reason, detail = self._classification(gt_fact, candidate)
            matches.append(
                {
                    "ground_truth_fact_id": gt_fact["id"],
                    "generated_fact_id": candidate["id"],
                    "classification": classification,
                    "section_score": 1 if gt_fact["section"] == candidate["section"] else 0,
                    "detail_score": detail,
                    "reason": reason,
                }
            )

        transcript_concepts = {fact["concept"] for fact in transcript_facts}
        unsupported = []
        for fact in gen_facts:
            if fact["id"] in used:
                continue
            supported = fact["concept"] in transcript_concepts
            unsupported.append(
                {
                    "generated_fact_id": fact["id"],
                    "classification": "SUPPORTED_BUT_ABSENT_FROM_GT" if supported else "UNSUPPORTED",
                    "reason": "Supported by transcript." if supported else "Unsupported generated fact.",
                }
            )

        return {
            "ground_truth_facts": gt_facts,
            "generated_facts": gen_facts,
            "fact_matches": matches,
            "unsupported_facts": unsupported,
            "clinical_error_events": self._clinical_error_events(matches, gt_facts, gen_facts),
            "judge_summary": "Single-call fake judgment.",
        }

    @staticmethod
    def _between(text, start, end):
        return text.split(start, 1)[1].split(end, 1)[0].strip()

    def _facts(self, note, prefix):
        lowered = note.lower()
        if "not applicable" in lowered and len(lowered.split()) <= 4:
            return []
        facts = []

        def fact(**kwargs):
            index = len(facts) + 1
            base = {
                "id": f"{prefix}_{index:04d}",
                "section": kwargs.pop("section", self._section(note)),
                "concept": kwargs.pop("concept"),
                "value": kwargs.pop("value", "present"),
                "status": kwargs.pop("status", "present"),
                "negation": kwargs.pop("negation", False),
                "evidence_text": kwargs.pop("evidence_text", note.strip()),
            }
            base.update(kwargs)
            facts.append(base)

        if "rhinorrhea" in lowered or "nasal discharge" in lowered:
            fact(concept="nasal discharge", duration="2 days" if "2" in lowered or "two" in lowered else None)
        if "no cough" in lowered:
            fact(concept="cough", value="absent", status="absent", negation=True, is_safety_relevant=True)
        elif "cough" in lowered:
            fact(concept="cough", source_or_speaker="patient" if "patient" in lowered else None)
        if "fucidin" in lowered:
            fact(
                concept="fucidin medication",
                medication_name="Fucidin",
                frequency="twice daily" if "twice daily" in lowered else None,
                duration="1 week" if "one week" in lowered else None,
                status="current" if "already" in lowered else "planned",
                temporality="current/prior" if "already" in lowered else "planned",
                is_safety_relevant=True,
            )
        return facts

    @staticmethod
    def _section(note):
        for section in ("Problem List", "Subjective", "Objective", "Assessment", "Plan"):
            if note.strip().startswith(f"{section}:"):
                return section
        return "Subjective"

    @staticmethod
    def _classification(gt_fact, gen_fact):
        if gt_fact.get("negation") != gen_fact.get("negation"):
            return "CONTRADICTION", "negation changed; presence/absence changed", 0
        if gt_fact.get("status") != gen_fact.get("status"):
            return "INCORRECT", "clinical status or temporality changed", 0.5
        missing = [
            field
            for field in ("frequency", "duration", "source_or_speaker", "certainty")
            if gt_fact.get(field) and not gen_fact.get(field)
        ]
        if missing:
            return "PARTIAL", "; ".join(f"{field} missing" for field in missing), 0.5
        return "CORRECT", "LLM semantic match.", 1

    @staticmethod
    def _clinical_error_events(matches, gt_facts, gen_facts):
        events = []
        for index, match in enumerate(matches, start=1):
            reason = match["reason"].lower()
            if match["classification"] == "CONTRADICTION" and "negation" in reason:
                events.append(
                    {
                        "type": "NEGATION_ERROR",
                        "severity": "MODERATE",
                        "ground_truth_fact_id": match["ground_truth_fact_id"],
                        "generated_fact_id": match["generated_fact_id"],
                        "error_group_id": f"negation_{index}",
                        "secondary_labels": ["CONTRADICTION"],
                        "is_safety_relevant": True,
                        "clinical_consequence": "Negation changed the documented clinical meaning.",
                        "reason": match["reason"],
                        "evidence_text": match["reason"],
                    }
                )
            if "fucidin" in reason or "frequency missing" in reason or "duration missing" in reason:
                events.append(
                    {
                        "type": "MEDICATION_DETAIL_OMISSION",
                        "severity": "MODERATE",
                        "ground_truth_fact_id": match["ground_truth_fact_id"],
                        "generated_fact_id": match["generated_fact_id"],
                        "error_group_id": f"medication_detail_{index}",
                        "secondary_labels": [],
                        "is_safety_relevant": True,
                        "clinical_consequence": "Medication details were omitted.",
                        "reason": match["reason"],
                        "evidence_text": match["reason"],
                    }
                )
        return events


class StatinAnemiaRegressionLLMClient:
    def generate_json(self, *, system_prompt, user_prompt, model, temperature):
        return {
            "ground_truth_facts": [
                {
                    "id": "gt_0001",
                    "section": "Assessment",
                    "concept": "hypercholesterolemia status",
                    "value": "elevated despite high-dose statin therapy",
                    "status": "abnormal",
                    "negation": False,
                    "evidence_text": "Hypercholesterolemia remains elevated despite high-dose statin therapy.",
                    "is_safety_relevant": True,
                },
                {
                    "id": "gt_0002",
                    "section": "Plan",
                    "concept": "statin dose plan",
                    "value": "reduce statin from 40 mg to 10 mg daily",
                    "status": "planned",
                    "negation": False,
                    "dose": "40 mg to 10 mg",
                    "medication_name": "statin",
                    "evidence_text": "Reduce statin from 40 mg to 10 mg daily.",
                    "is_safety_relevant": True,
                },
                {
                    "id": "gt_0003",
                    "section": "Objective",
                    "concept": "anemia",
                    "value": "absent",
                    "status": "absent",
                    "negation": True,
                    "evidence_text": "No anemia noted.",
                    "is_safety_relevant": False,
                },
            ],
            "generated_facts": [
                {
                    "id": "gen_0001",
                    "section": "Assessment",
                    "concept": "hyperlipidemia status",
                    "value": "improving on statin therapy",
                    "status": "present",
                    "negation": False,
                    "evidence_text": "Hyperlipidemia is improving on statin therapy.",
                    "is_safety_relevant": True,
                },
                {
                    "id": "gen_0002",
                    "section": "Plan",
                    "concept": "statin dose plan",
                    "value": "continue statin therapy at bedtime",
                    "status": "planned",
                    "negation": False,
                    "medication_name": "statin",
                    "evidence_text": "Continue statin therapy at bedtime.",
                    "is_safety_relevant": True,
                },
            ],
            "fact_matches": [
                {
                    "ground_truth_fact_id": "gt_0001",
                    "generated_fact_id": "gen_0001",
                    "classification": "CONTRADICTION",
                    "section_score": 1.0,
                    "detail_score": 0.5,
                    "reason": "Cholesterol status changed from elevated despite high-dose statin to improving on statin therapy.",
                },
                {
                    "ground_truth_fact_id": "gt_0002",
                    "generated_fact_id": "gen_0002",
                    "classification": "INCORRECT",
                    "section_score": 1.0,
                    "detail_score": 0.4,
                    "reason": "Medication plan and dose mismatch: reduce statin from 40 mg to 10 mg daily versus continue statin therapy.",
                },
                {
                    "ground_truth_fact_id": "gt_0003",
                    "generated_fact_id": None,
                    "classification": "MISSING",
                    "section_score": 0.0,
                    "detail_score": 0.0,
                    "reason": "Absence of anemia not mentioned; no clear safety consequence.",
                },
            ],
            "unsupported_facts": [],
            "clinical_error_events": [
                {
                    "type": "MEDICATION_ERROR",
                    "severity": "HIGH",
                    "ground_truth_fact_id": "gt_0002",
                    "generated_fact_id": "gen_0002",
                    "error_group_id": "statin_gt_0002",
                    "secondary_labels": ["DOSE_ERROR"],
                    "is_safety_relevant": True,
                    "clinical_consequence": "The generated note changes the documented medication plan.",
                    "reason": "Medication plan and dose mismatch: reduce statin from 40 mg to 10 mg daily versus continue statin therapy.",
                    "evidence_text": "Reduce statin from 40 mg to 10 mg daily. Continue statin therapy at bedtime.",
                }
            ],
            "judge_summary": "Regression case with statin plan mismatch and ordinary missing anemia fact.",
        }


class LLMSeverityJudgmentClient:
    def generate_json(self, *, system_prompt, user_prompt, model, temperature):
        return {
            "ground_truth_facts": [
                {
                    "id": "gt_0001",
                    "section": "Plan",
                    "concept": "drug allergy",
                    "value": "penicillin allergy",
                    "status": "present",
                    "negation": False,
                    "evidence_text": "Patient has penicillin allergy.",
                    "is_safety_relevant": True,
                }
            ],
            "generated_facts": [
                {
                    "id": "gen_0001",
                    "section": "Plan",
                    "concept": "drug allergy",
                    "value": "no known drug allergies",
                    "status": "absent",
                    "negation": True,
                    "evidence_text": "No known drug allergies.",
                    "is_safety_relevant": True,
                }
            ],
            "fact_matches": [
                {
                    "ground_truth_fact_id": "gt_0001",
                    "generated_fact_id": "gen_0001",
                    "classification": "CONTRADICTION",
                    "section_score": 1.0,
                    "detail_score": 0.0,
                    "reason": "Generated note reverses documented penicillin allergy.",
                }
            ],
            "unsupported_facts": [],
            "clinical_error_events": [
                {
                    "type": "ALLERGY_ERROR",
                    "severity": "CRITICAL",
                    "ground_truth_fact_id": "gt_0001",
                    "generated_fact_id": "gen_0001",
                    "error_group_id": "allergy_gt_0001",
                    "secondary_labels": ["NEGATION_ERROR"],
                    "is_safety_relevant": True,
                    "clinical_consequence": "Could expose the patient to an unsafe medication.",
                    "reason": "Generated note reverses documented penicillin allergy.",
                    "evidence_text": "Patient has penicillin allergy. No known drug allergies.",
                }
            ],
            "judge_summary": "LLM-provided clinical event severity.",
        }


def evaluator(client=None, config=None):
    return ClinicalNoteEvaluator(
        config=config or MetricConfig(),
        llm_client=client or SingleCallFakeLLMClient(),
    )


def test_evaluator_requires_llm_client():
    with pytest.raises(ValueError, match="requires an LLM client"):
        ClinicalNoteEvaluator()


def test_clinical_fact_normalizes_llm_status_variants():
    fact = ClinicalFact.model_validate(
        {
            "id": "x",
            "section": "problem_list",
            "concept": "diabetes mellitus",
            "status": "active",
            "evidence_text": "Diabetes mellitus",
        }
    )

    assert fact.section == "Problem List"
    assert fact.status == "present"


def test_single_call_judge_paraphrase_match():
    client = SingleCallFakeLLMClient()
    result = evaluator(client).evaluate(
        ground_truth_note="Nasal discharge for 2 days.",
        generated_note="Two-day history of rhinorrhea.",
    )

    assert client.calls == 1
    assert result.counts.correct == 1
    assert result.final_score == 100


def test_single_call_judge_detects_negation_error():
    result = evaluator().evaluate(
        ground_truth_note="No cough.",
        generated_note="Patient reports cough.",
    )

    assert result.counts.contradictions == 1
    assert any(event.type == "NEGATION_ERROR" for event in result.clinical_error_events)


def test_single_call_judge_detects_missing_medication_detail():
    result = evaluator().evaluate(
        ground_truth_note="Plan: Apply Fucidin twice daily for one week.",
        generated_note="Plan: Apply Fucidin.",
    )

    assert result.counts.partial == 1
    assert result.scores.correctness < 100


def test_single_call_judge_supports_transcript_absent_from_gt():
    result = evaluator().evaluate(
        ground_truth_note="Nasal discharge.",
        generated_note="Nasal discharge with cough.",
        transcript="Mother says the child has nasal discharge and cough.",
    )

    assert result.counts.supported_but_absent_from_gt == 1
    assert result.counts.unsupported == 0


def test_empty_section_has_no_facts():
    result = evaluator().evaluate(
        ground_truth_note="Objective: Not applicable",
        generated_note="Objective: Not applicable",
    )

    assert result.counts.ground_truth_facts == 0
    assert result.counts.generated_facts == 0
    assert result.final_score == 100


def test_configurable_weights_still_apply_after_llm_judgment():
    config = MetricConfig(
        weights={
            "completeness": 1,
            "correctness": 0,
            "supported_content": 0,
            "section_placement": 0,
        }
    )
    result = evaluator(config=config).evaluate(
        ground_truth_note="No cough.",
        generated_note="Patient reports cough.",
    )

    assert result.fidelity_score == 0
    assert result.final_score == 0


def test_single_call_judgment_retries_when_generated_fact_is_omitted():
    client = SingleCallFakeLLMClient(omit_extra_once=True)
    result = evaluator(client=client, config=MetricConfig(llm_retry_attempts=1)).evaluate(
        ground_truth_note="Nasal discharge.",
        generated_note="Nasal discharge. Patient reports cough.",
    )

    assert client.calls == 2
    assert result.counts.unsupported == 1


def test_statin_anemia_regression_keeps_error_profile_separate_from_score():
    result = evaluator(client=StatinAnemiaRegressionLLMClient()).evaluate(
        ground_truth_note=(
            "Assessment: Hypercholesterolemia remains elevated despite high-dose statin therapy.\n"
            "Plan: Reduce statin from 40 mg to 10 mg daily.\n"
            "Objective: No anemia noted."
        ),
        generated_note=(
            "Assessment: Hyperlipidemia is improving on statin therapy.\n"
            "Plan: Continue statin therapy at bedtime."
        ),
    )

    anemia_match = next(match for match in result.fact_matches if match.ground_truth_fact.concept == "anemia")
    assert anemia_match.classification == "MISSING"
    assert not any(
        event.type == "CRITICAL_OMISSION"
        and event.ground_truth_fact
        and event.ground_truth_fact.concept == "anemia"
        for event in result.clinical_error_events
    )

    statin_events = [
        event
        for event in result.clinical_error_events
        if event.error_group_id.startswith("statin_")
    ]
    assert len(statin_events) == 1
    assert statin_events[0].type in {"DOSE_ERROR", "MEDICATION_ERROR", "MEDICATION_DETAIL_OMISSION"}
    assert statin_events[0].severity in {"MODERATE", "HIGH", "CRITICAL"}
    assert result.final_score == result.fidelity_score
    assert result.overall_fidelity_score == result.fidelity_score
    assert not hasattr(result, "total_penalty")
    assert not hasattr(statin_events[0], "penalty")


def test_evaluator_preserves_llm_provided_clinical_event_severity():
    result = evaluator(client=LLMSeverityJudgmentClient()).evaluate(
        ground_truth_note="Plan: Patient has penicillin allergy.",
        generated_note="Plan: No known drug allergies.",
    )

    assert len(result.clinical_error_events) == 1
    event = result.clinical_error_events[0]
    assert event.type == "ALLERGY_ERROR"
    assert event.severity == "CRITICAL"
    assert event.reason == "Generated note reverses documented penicillin allergy."
