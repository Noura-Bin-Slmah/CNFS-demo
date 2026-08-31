"""Main CNFS evaluator."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

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
    SectionPlacementIssue,
)
from clinical_note_metric.prompts import FACT_EXTRACTION_PROMPT, MATCH_AND_CLASSIFY_PROMPT
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
        self.section_evaluator = SectionEvaluator(self.config)

    def evaluate(
        self,
        *,
        ground_truth_note: str,
        generated_note: str,
        transcript: str | None = None,
    ) -> CNFSResult:
        """Run the complete CNFS pipeline and return a structured result."""

        with ThreadPoolExecutor(max_workers=2) as pool:
            gt_future = pool.submit(
                self._extract_facts,
                note_text=ground_truth_note,
                note_role="ground-truth",
                note_role_title="Ground-truth",
                id_prefix="gt",
            )
            gen_future = pool.submit(
                self._extract_facts,
                note_text=generated_note,
                note_role="generated",
                note_role_title="Generated",
                id_prefix="gen",
            )
            gt_facts = gt_future.result()
            generated_facts = gen_future.result()

        matches, extras, clinical_error_events, section_placement_issues = self._match_and_classify(
            ground_truth_note=ground_truth_note,
            generated_note=generated_note,
            transcript=transcript,
            gt_facts=gt_facts,
            generated_facts=generated_facts,
        )
        scores = self.scorer.dimension_scores(matches, extras, len(generated_facts))
        fidelity_score = self.scorer.base_score(scores)
        final_score = fidelity_score

        counts = self._counts(
            gt_facts, generated_facts, matches, extras, clinical_error_events, section_placement_issues
        )
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
            f"{counts.clinical_error_events} clinical error events."
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
            section_placement_issues=section_placement_issues,
            transcript_used=transcript is not None,
            limitations=limitations,
            summary=summary,
            raw_judge_reasoning_summary=[
                match.reason for match in matches if match.reason
            ]
            + [extra.reason for extra in extras],
        )

    def _extract_facts(
        self,
        *,
        note_text: str,
        note_role: str,
        note_role_title: str,
        id_prefix: str,
    ) -> list[ClinicalFact]:
        prompt = FACT_EXTRACTION_PROMPT.format(
            prompt_version=self.config.prompt_version,
            note_role=note_role,
            note_role_title=note_role_title,
            id_prefix=id_prefix,
            note_text=note_text,
        )

        last_error: ValueError | None = None
        repair_instruction = ""
        for _attempt in range(self.config.llm_retry_attempts + 1):
            payload = self.judge.ask_json(prompt + repair_instruction)
            try:
                return self._parse_extraction_payload(payload, id_prefix=id_prefix)
            except ValueError as exc:
                last_error = exc
                repair_instruction = (
                    "\n\nYour previous JSON was incomplete or invalid. "
                    f"Fix this exact issue: {exc}. Return the full corrected JSON again."
                )
        raise ValueError(
            f"Fact extraction for the {note_role} note remained invalid after retries."
        ) from last_error

    @staticmethod
    def _parse_extraction_payload(payload: dict, *, id_prefix: str) -> list[ClinicalFact]:
        raw_facts = payload.get("facts")
        if not isinstance(raw_facts, list):
            raise ValueError("extraction output is missing a 'facts' list")
        return [
            ClinicalFact.model_validate({**raw, "id": raw.get("id") or f"{id_prefix}_{index:04d}"})
            for index, raw in enumerate(raw_facts, start=1)
        ]

    def _match_and_classify(
        self,
        *,
        ground_truth_note: str,
        generated_note: str,
        transcript: str | None,
        gt_facts: list[ClinicalFact],
        generated_facts: list[ClinicalFact],
    ) -> tuple[list[FactMatch], list[ExtraGeneratedFact], list[ClinicalErrorEvent], list[SectionPlacementIssue]]:
        gt_by_id = {fact.id: fact for fact in gt_facts}
        gen_by_id = {fact.id: fact for fact in generated_facts}

        def _compact(facts: list[ClinicalFact]) -> list[dict]:
            return [
                {"id": f.id, "section": f.section, "concept": f.concept, "evidence_text": f.evidence_text}
                for f in facts
            ]

        prompt = MATCH_AND_CLASSIFY_PROMPT.format(
            prompt_version=self.config.prompt_version,
            ground_truth_note=ground_truth_note,
            generated_note=generated_note,
            transcript=transcript or "None provided",
            ground_truth_facts_json=json.dumps(_compact(gt_facts), indent=2),
            generated_facts_json=json.dumps(_compact(generated_facts), indent=2),
        )

        last_error: ValueError | None = None
        repair_instruction = ""
        for _attempt in range(self.config.llm_retry_attempts + 1):
            payload = self.judge.ask_json(prompt + repair_instruction)
            try:
                return self._parse_match_payload(payload, gt_by_id=gt_by_id, gen_by_id=gen_by_id)
            except ValueError as exc:
                last_error = exc
                repair_instruction = (
                    "\n\nYour previous JSON was incomplete or invalid for CNFS. "
                    f"Fix this exact issue: {exc}. Return the full corrected JSON again."
                )
        raise ValueError("Match-and-classify judgment remained incomplete after retries.") from last_error

    @staticmethod
    def _parse_match_payload(
        payload: dict,
        *,
        gt_by_id: dict[str, ClinicalFact],
        gen_by_id: dict[str, ClinicalFact],
    ) -> tuple[list[FactMatch], list[ExtraGeneratedFact], list[ClinicalErrorEvent], list[SectionPlacementIssue]]:
        # Two invariants below (MISSING must not carry a generated fact; a match must
        # stay within one section) are things the LLM sometimes gets wrong even after a
        # repair retry — rejecting outright risked exhausting retries and failing the
        # whole request. Sanitizing in place (drop the bad reference, downgrade to
        # MISSING) keeps the evaluation resilient; any generated fact this orphans is
        # swept into extras below rather than silently vanishing from the coverage count.
        matches: list[FactMatch] = []
        used_generated_ids: set[str] = set()
        for raw_match in payload.get("fact_matches", []):
            gt_fact = gt_by_id.get(raw_match.get("ground_truth_fact_id"))
            if gt_fact is None:
                continue
            generated_id = raw_match.get("generated_fact_id")
            gen_fact = gen_by_id.get(generated_id) if generated_id else None
            classification = MatchClassification(raw_match.get("classification", "MISSING"))

            if classification == MatchClassification.MISSING and gen_fact is not None:
                LOGGER.warning(
                    "Dropping generated_fact_id %s attached to a MISSING match for %s.",
                    gen_fact.id, gt_fact.id,
                )
                gen_fact = None
            elif gen_fact is not None and gen_fact.section != gt_fact.section:
                LOGGER.warning(
                    "Match for %s (%s) named generated fact %s in a different section "
                    "(%s) — discarding the cross-section match; %s is now MISSING.",
                    gt_fact.id, gt_fact.section, gen_fact.id, gen_fact.section, gt_fact.id,
                )
                gen_fact = None
                classification = MatchClassification.MISSING

            if classification != MatchClassification.MISSING and gen_fact is None:
                raise ValueError(
                    f"fact_matches entry for {gt_fact.id} has classification {classification} "
                    "but no valid generated_fact_id — a non-MISSING classification requires an "
                    "actual matched generated fact in the same section."
                )
            if gen_fact:
                used_generated_ids.add(gen_fact.id)
            matches.append(
                FactMatch(
                    ground_truth_fact=gt_fact,
                    generated_fact=gen_fact,
                    classification=classification,
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
            raise ValueError(f"match-and-classify output omitted ground-truth facts: {sorted(missing_gt_ids)}")

        # Any generated fact orphaned by the sanitization above (or simply never
        # mentioned by the LLM) is defensively treated as unsupported rather than
        # raising — conservative (grants no undue credit) and keeps the request from
        # failing over a coverage gap the LLM should have closed itself.
        for generated_id in sorted(set(gen_by_id) - used_generated_ids):
            gen_fact = gen_by_id[generated_id]
            LOGGER.warning(
                "Generated fact %s was left unaccounted for; defaulting it to unsupported.",
                gen_fact.id,
            )
            extras.append(
                ExtraGeneratedFact(
                    generated_fact=gen_fact,
                    classification=GeneratedFactClassification.UNSUPPORTED,
                    reason="Automatically marked unsupported — the judge did not account for this fact.",
                )
            )

        clinical_error_events = ClinicalNoteEvaluator._parse_clinical_error_events(
            payload.get("clinical_error_events", []),
            gt_by_id,
            gen_by_id,
        )

        section_placement_issues: list[SectionPlacementIssue] = []
        for raw_issue in payload.get("section_placement_issues", []):
            gt_fact = gt_by_id.get(raw_issue.get("ground_truth_fact_id"))
            gen_fact = gen_by_id.get(raw_issue.get("generated_fact_id"))
            if gt_fact is None or gen_fact is None:
                continue
            section_placement_issues.append(
                SectionPlacementIssue(
                    ground_truth_fact=gt_fact,
                    generated_fact=gen_fact,
                    reason=str(raw_issue.get("reason", "LLM identified a section placement mismatch.")),
                )
            )

        return matches, extras, clinical_error_events, section_placement_issues

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
    def _counts(
        gt_facts, generated_facts, matches, extras, clinical_error_events, section_placement_issues
    ) -> CountBreakdown:
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
            section_placement_issues=len(section_placement_issues),
        )
