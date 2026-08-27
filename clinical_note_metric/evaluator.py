"""Main CNFS evaluator."""

from __future__ import annotations

import logging

from clinical_note_metric.config import MetricConfig
from clinical_note_metric.judge import JsonLLMJudge, LLMClient
from clinical_note_metric.models import (
    CNFSResult,
    ClinicalErrorEvent,
    ClinicalErrorEventType,
    ClinicalFact,
    CountBreakdown,
    ExtraGeneratedFact,
    FactMatch,
    GeneratedFactClassification,
    MatchClassification,
)
from clinical_note_metric.prompts import SINGLE_CALL_EVALUATION_PROMPT
from clinical_note_metric.scoring import CNFSScorer
from clinical_note_metric.section_evaluator import SectionEvaluator

LOGGER = logging.getLogger(__name__)


class ClinicalNoteEvaluator:
    """Evaluates generated medical notes against a ground truth note and transcript."""

    def __init__(
        self,
        *,
        model: str | None = None,
        config: MetricConfig | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.config = config or MetricConfig(model=model)
        if model is not None and config is None:
            self.config.model = model
        if llm_client is None:
            raise ValueError(
                "ClinicalNoteEvaluator requires an LLM client. "
                "Pass OpenAIJudgeClient() or another LLMClient implementation."
            )
        self.config.use_llm_extraction = True
        self.config.use_llm_matching = True
        self.judge = JsonLLMJudge(llm_client, self.config)
        self.scorer = CNFSScorer(self.config)
        self.section_evaluator = SectionEvaluator()

    def evaluate(
        self,
        *,
        ground_truth_note: str,
        generated_note: str,
        transcript: str | None = None,
    ) -> CNFSResult:
        """Run the complete CNFS pipeline and return a structured result."""

        gt_facts, generated_facts, matches, extras, clinical_error_events, judge_summary = self._single_call_judgment(
            ground_truth_note=ground_truth_note,
            generated_note=generated_note,
            transcript=transcript,
        )
        scores = self.scorer.dimension_scores(matches, extras, len(generated_facts))
        fidelity_score = self.scorer.base_score(scores)
        final_score = fidelity_score

        counts = self._counts(gt_facts, generated_facts, matches, extras, clinical_error_events)
        section_scores = self.section_evaluator.evaluate(matches, extras)
        limitations = []
        if transcript is None:
            limitations.append(
                "No transcript was provided; unmatched generated facts were evaluated only against the ground truth."
            )

        summary = (
            f"CNFS {final_score:.1f}/100 from {len(gt_facts)} ground-truth facts and "
            f"{len(generated_facts)} generated facts; {counts.unsupported} unsupported, "
            f"{counts.missing} missing, {counts.contradictions} contradictions, "
            f"{counts.clinical_error_events} clinical error events. "
            f"{judge_summary}".strip()
        )

        return CNFSResult(
            metric_version=self.config.metric_version,
            final_score=final_score,
            overall_fidelity_score=fidelity_score,
            fidelity_score=fidelity_score,
            scores=scores,
            counts=counts,
            section_scores=section_scores,
            fact_matches=matches,
            unsupported_facts=extras,
            clinical_error_events=clinical_error_events,
            transcript_used=transcript is not None,
            limitations=limitations,
            summary=summary,
            raw_judge_reasoning_summary=[
                match.reason for match in matches if match.reason
            ]
            + [extra.reason for extra in extras],
        )

    def _single_call_judgment(
        self,
        *,
        ground_truth_note: str,
        generated_note: str,
        transcript: str | None,
    ) -> tuple[
        list[ClinicalFact],
        list[ClinicalFact],
        list[FactMatch],
        list[ExtraGeneratedFact],
        list[ClinicalErrorEvent],
        str,
    ]:
        prompt = SINGLE_CALL_EVALUATION_PROMPT.format(
            prompt_version=self.config.prompt_version,
            ground_truth_note=ground_truth_note,
            generated_note=generated_note,
            transcript=transcript or "None provided",
        )

        last_error: ValueError | None = None
        repair_instruction = ""
        for _attempt in range(self.config.llm_retry_attempts + 1):
            payload = self.judge.ask_json(prompt + repair_instruction)
            try:
                return self._parse_single_call_payload(payload)
            except ValueError as exc:
                last_error = exc
                repair_instruction = (
                    "\n\nYour previous JSON was incomplete or invalid for CNFS. "
                    f"Fix this exact issue: {exc}. Return the full corrected JSON again."
                )
        raise ValueError("Single-call LLM judgment remained incomplete after retries.") from last_error

    @staticmethod
    def _parse_single_call_payload(
        payload: dict,
    ) -> tuple[
        list[ClinicalFact],
        list[ClinicalFact],
        list[FactMatch],
        list[ExtraGeneratedFact],
        list[ClinicalErrorEvent],
        str,
    ]:
        gt_facts = [
            ClinicalFact.model_validate({**raw, "id": raw.get("id") or f"gt_{index:04d}"})
            for index, raw in enumerate(payload.get("ground_truth_facts", []), start=1)
        ]
        generated_facts = [
            ClinicalFact.model_validate({**raw, "id": raw.get("id") or f"gen_{index:04d}"})
            for index, raw in enumerate(payload.get("generated_facts", []), start=1)
        ]
        gt_by_id = {fact.id: fact for fact in gt_facts}
        gen_by_id = {fact.id: fact for fact in generated_facts}

        matches: list[FactMatch] = []
        used_generated_ids: set[str] = set()
        for raw_match in payload.get("fact_matches", []):
            gt_fact = gt_by_id.get(raw_match.get("ground_truth_fact_id"))
            if gt_fact is None:
                continue
            generated_id = raw_match.get("generated_fact_id")
            gen_fact = gen_by_id.get(generated_id) if generated_id else None
            if gen_fact:
                used_generated_ids.add(gen_fact.id)
            matches.append(
                FactMatch(
                    ground_truth_fact=gt_fact,
                    generated_fact=gen_fact,
                    classification=MatchClassification(raw_match.get("classification", "MISSING")),
                    section_score=float(raw_match.get("section_score", 0.0 if gen_fact is None else 1.0)),
                    reason=str(raw_match.get("reason", "LLM semantic judgment.")),
                )
            )

        extras: list[ExtraGeneratedFact] = []
        for raw_extra in payload.get("unsupported_facts", []):
            gen_fact = gen_by_id.get(raw_extra.get("generated_fact_id"))
            if gen_fact is None:
                continue
            used_generated_ids.add(gen_fact.id)
            extras.append(
                ExtraGeneratedFact(
                    generated_fact=gen_fact,
                    classification=GeneratedFactClassification(raw_extra.get("classification", "UNSUPPORTED")),
                    reason=str(raw_extra.get("reason", "LLM extra-fact judgment.")),
                )
            )

        missing_gt_ids = set(gt_by_id) - {match.ground_truth_fact.id for match in matches}
        if missing_gt_ids:
            raise ValueError(f"single-call output omitted ground-truth facts: {sorted(missing_gt_ids)}")

        unaccounted_generated_ids = set(gen_by_id) - used_generated_ids
        if unaccounted_generated_ids:
            raise ValueError(
                "single-call output omitted generated facts: "
                f"{sorted(unaccounted_generated_ids)}"
            )

        clinical_error_events = ClinicalNoteEvaluator._parse_clinical_error_events(
            payload.get("clinical_error_events", []),
            gt_by_id,
            gen_by_id,
        )

        return (
            gt_facts,
            generated_facts,
            matches,
            extras,
            clinical_error_events,
            str(payload.get("judge_summary", "")).strip(),
        )

    @staticmethod
    def _parse_clinical_error_events(
        raw_events: list[dict],
        gt_by_id: dict[str, ClinicalFact],
        gen_by_id: dict[str, ClinicalFact],
    ) -> list[ClinicalErrorEvent]:
        events: list[ClinicalErrorEvent] = []
        for index, raw_event in enumerate(raw_events, start=1):
            gt_fact = gt_by_id.get(raw_event.get("ground_truth_fact_id") or "")
            gen_fact = gen_by_id.get(raw_event.get("generated_fact_id") or "")
            event_type = ClinicalErrorEventType(raw_event.get("type", "MISSING_DETAIL"))
            reason = str(raw_event.get("reason", "")).strip()
            group_id = str(raw_event.get("error_group_id", "")).strip()
            if not group_id:
                group_source = gt_fact or gen_fact
                group_id = f"{event_type.value.lower()}_{group_source.id if group_source else index}"
            events.append(
                ClinicalErrorEvent(
                    type=event_type,
                    severity=raw_event.get("severity", "LOW"),
                    error_group_id=group_id,
                    primary_error=event_type,
                    secondary_labels=[
                        str(label)
                        for label in raw_event.get("secondary_labels", [])
                        if str(label) != event_type.value
                    ],
                    ground_truth_fact=gt_fact,
                    generated_fact=gen_fact,
                    reason=reason or "LLM identified a clinical documentation fidelity event.",
                )
            )
        return ClinicalNoteEvaluator._deduplicate_clinical_error_events(events)

    @staticmethod
    def _deduplicate_clinical_error_events(events: list[ClinicalErrorEvent]) -> list[ClinicalErrorEvent]:
        severity_rank = {"NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}
        best_by_group: dict[str, ClinicalErrorEvent] = {}
        labels_by_group: dict[str, set[str]] = {}
        for event in events:
            labels_by_group.setdefault(event.error_group_id, set()).add(event.type)
            current = best_by_group.get(event.error_group_id)
            if current is None or severity_rank[event.severity] > severity_rank[current.severity]:
                best_by_group[event.error_group_id] = event

        deduped: list[ClinicalErrorEvent] = []
        for group_id, event in best_by_group.items():
            secondary = set(event.secondary_labels)
            secondary.update(label for label in labels_by_group[group_id] if label != event.type)
            deduped.append(event.model_copy(update={"secondary_labels": sorted(secondary)}))
        return deduped

    @staticmethod
    def _counts(gt_facts, generated_facts, matches, extras, clinical_error_events) -> CountBreakdown:
        return CountBreakdown(
            ground_truth_facts=len(gt_facts),
            generated_facts=len(generated_facts),
            correct=sum(1 for m in matches if m.classification == MatchClassification.CORRECT),
            partial=sum(1 for m in matches if m.classification == MatchClassification.PARTIAL),
            incorrect=sum(1 for m in matches if m.classification == MatchClassification.INCORRECT),
            contradictions=sum(
                1 for m in matches if m.classification == MatchClassification.CONTRADICTION
            ),
            missing=sum(1 for m in matches if m.classification == MatchClassification.MISSING),
            unsupported=sum(
                1 for extra in extras if extra.classification == GeneratedFactClassification.UNSUPPORTED
            ),
            supported_but_absent_from_gt=sum(
                1
                for extra in extras
                if extra.classification
                == GeneratedFactClassification.SUPPORTED_BUT_ABSENT_FROM_GT
            ),
            clinical_error_events=len(clinical_error_events),
        )
