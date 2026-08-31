"""Local dev server for the CNFS demo.

Serves the static frontend and exposes POST /api/evaluate, which runs the
real ClinicalNoteEvaluator (OpenAI-backed) against the notes submitted from
the page.
"""

from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from clinical_note_metric import ClinicalNoteEvaluator, MetricConfig, OpenAIJudgeClient
from run_cnfs import load_env_file, load_note_text, validate_openai_key

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
DEFAULT_MODEL = os.getenv("CNFS_OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")


def evaluate_notes(ground_truth_note: str, generated_note: str, transcript: str | None) -> dict:
    evaluator = ClinicalNoteEvaluator(
        config=MetricConfig(model=DEFAULT_MODEL, temperature=0.0),
        llm_client=OpenAIJudgeClient(),
    )
    result = evaluator.evaluate(
        ground_truth_note=ground_truth_note,
        generated_note=generated_note,
        transcript=transcript,
    )

    fact_rows = [
        {
            "classification": match.classification,
            "concept": match.ground_truth_fact.concept,
            "groundTruthSection": match.ground_truth_fact.section,
            "generatedSection": match.generated_fact.section if match.generated_fact else None,
            "groundTruth": match.ground_truth_fact.evidence_text,
            "generated": match.generated_fact.evidence_text if match.generated_fact else None,
            "reason": match.reason,
        }
        for match in result.fact_matches
    ] + [
        {
            "classification": "UNSUPPORTED",
            "concept": extra.generated_fact.concept,
            "groundTruthSection": None,
            "generatedSection": extra.generated_fact.section,
            "groundTruth": None,
            "generated": extra.generated_fact.evidence_text,
            "reason": extra.reason,
        }
        for extra in result.unsupported_facts
        if extra.classification == "UNSUPPORTED"
    ]

    return {
        "overallScore": result.overall_fidelity_score,
        "summary": result.summary,
        "scores": {
            "completeness": result.scores.completeness,
            "correctness": result.scores.correctness,
            "supportedContent": result.scores.supported_content,
        },
        "weights": dict(evaluator.config.weights),
        "sectionScores": {
            section: {
                "factCount": section_score.fact_count,
                "generatedFactCount": section_score.generated_fact_count,
                "completeness": section_score.completeness,
                "correctness": section_score.correctness,
                "supportedContent": section_score.supported_content,
                "overall": section_score.overall,
                "correctCount": section_score.correct_fact_count,
                "partialCount": section_score.partial_fact_count,
                "missingCount": section_score.missing_fact_count,
                "incorrectCount": section_score.incorrect_fact_count,
                "contradictionCount": section_score.contradiction_count,
                "unsupportedCount": section_score.unsupported_fact_count,
            }
            for section, section_score in result.section_scores.items()
        },
        "counts": {
            "groundTruthFacts": result.counts.ground_truth_facts,
            "generatedFacts": result.counts.generated_facts,
            "correct": result.counts.correct,
            "partial": result.counts.partial,
            "missing": result.counts.missing,
            "incorrect": result.counts.incorrect,
            "contradictions": result.counts.contradictions,
            "unsupported": result.counts.unsupported,
            "clinicalErrorEvents": result.counts.clinical_error_events,
            "sectionPlacementIssues": result.counts.section_placement_issues,
        },
        "events": [
            {"type": event.type, "severity": event.severity, "reason": event.reason}
            for event in result.clinical_error_events
        ],
        "sectionPlacementIssues": [
            {
                "concept": issue.ground_truth_fact.concept,
                "expectedSection": issue.ground_truth_fact.section,
                "actualSection": issue.generated_fact.section,
                "groundTruth": issue.ground_truth_fact.evidence_text,
                "generated": issue.generated_fact.evidence_text,
                "reason": issue.reason,
            }
            for issue in result.section_placement_issues
        ],
        "factMatches": fact_rows,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/evaluate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
            ground_truth_note = load_note_text(payload.get("groundTruth", ""))
            generated_note = load_note_text(payload.get("generated", ""))
            transcript = payload.get("transcript") or None
            self._send_json(200, evaluate_notes(ground_truth_note, generated_note, transcript))
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})


def main() -> None:
    load_env_file()
    validate_openai_key()
    port = int(os.getenv("PORT", os.getenv("CNFS_PORT", "8000")))
    host = os.getenv("CNFS_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"CNFS demo server running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
