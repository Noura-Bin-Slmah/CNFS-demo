"""Run a single Clinical Note Fidelity Score evaluation.

Edit the placeholder notes below for quick testing, or pass note files with
--ground-truth and --generated.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from clinical_note_metric import ClinicalNoteEvaluator, MetricConfig, OpenAIJudgeClient

GROUND_TRUTH_NOTE_PLACEHOLDER = """
{
  "problem_list": [
    "Diabetes mellitus",
    "Hypertension",
    "Hypercholesterolemia",
    "Foot pain"
  ],
  "subjective": "Patient reports persistent foot pain in multiple areas limiting movement, monitors blood sugar without a meter, and notes recent readings of 6.8 then 6.3. She mentions blood pressure readings, a recent value of 136/86, and a history of high cholesterol treated with a high-dose medication. Labs show hemoglobin 14, normal kidney function, and adequate vitamin D and B12 levels. No episodes of hypoglycemia or anemia are reported.",
  "objective": "Blood pressure 136/86 mmHg. Recent blood glucose 6.3 (units not specified). Hemoglobin 14 g/dL. Kidney function within normal limits. Vitamin D and vitamin B12 levels normal. No anemia noted.",
  "assessment": "Diabetes mellitus is currently controlled. Hypertension is controlled. Hypercholesterolemia remains elevated despite high-dose statin therapy. Foot pain is likely musculoskeletal in origin.",
  "plan": "Continue current diabetes regimen. Continue antihypertensive therapy. Reduce statin dose from 40 mg to 10 mg daily, taken each day preferably at bedtime. Follow up in three months."
}
"""

GENERATED_NOTE_PLACEHOLDER = """
{
  "problem_list": [
    "Diabetes mellitus",
    "Hypertension",
    "Hyperlipidemia",
    "Foot pain"
  ],
  "subjective": "Patient reports ongoing foot pain that affects movement. She checks her blood sugar and reports recent readings around 6.3. She also reports a recent blood pressure reading of 136/86. She has a history of elevated cholesterol and is taking cholesterol medication. She denies hypoglycemic episodes.",
  "objective": "Blood pressure is 136/86 mmHg. Blood glucose is 6.3. Hemoglobin is 14 g/dL. Kidney function is normal. Vitamin D and B12 are within normal limits.",
  "assessment": "Diabetes mellitus is controlled. Hypertension is controlled. Hyperlipidemia is improving on statin therapy. Foot pain is likely musculoskeletal.",
  "plan": "Continue diabetes medications. Continue blood pressure medications. Continue statin therapy at bedtime. Follow up in three months."
}
"""

TRANSCRIPT_PLACEHOLDER = None

PLACEHOLDER_KEYS = {
    "your_api_key_here",
    "your_actual_api_key",
    "your_actual_api_key_here",
}


def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file without extra dependencies."""

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def validate_openai_key() -> None:
    """Fail fast when the key is missing or still a placeholder."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before running this script.")
    if api_key.lower() in PLACEHOLDER_KEYS or "your_api" in api_key.lower():
        raise RuntimeError(
            "OPENAI_API_KEY is still a placeholder. Replace it in .env with your real OpenAI API key."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNFS for any ground-truth/generated note pair.")
    parser.add_argument(
        "--ground-truth",
        help="Path to the ground-truth note file. JSON section files and plain text are supported.",
    )
    parser.add_argument(
        "--generated",
        help="Path to the LLM-generated note file. JSON section files and plain text are supported.",
    )
    parser.add_argument(
        "--transcript",
        help="Optional path to a transcript file used only to classify generated additions.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CNFS_OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model name. Defaults to CNFS_OPENAI_MODEL or gpt-4.1-mini.",
    )
    return parser.parse_args()


def load_note_text(raw_text: str) -> str:
    """Load a note from JSON sections or plain text."""

    text = raw_text.strip()
    if not text:
        raise ValueError("Note text is empty.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(parsed, dict):
        return note_dict_to_text(parsed)
    if isinstance(parsed, str):
        return parsed
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def load_note_file(path: str) -> str:
    """Load a note from JSON sections or plain text."""

    raw_text = Path(path).read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Input file is empty: {path}")
    return load_note_text(raw_text)


def note_dict_to_text(note: dict[str, Any]) -> str:
    """Convert a note dictionary into CNFS section headers."""

    problem_list = note.get("problem_list", [])
    if isinstance(problem_list, list):
        problem_list_text = "\n".join(f"- {item}" for item in problem_list)
    else:
        problem_list_text = str(problem_list)

    sections = {
        "Problem List": problem_list_text,
        "Subjective": note.get("subjective", "Not applicable"),
        "Objective": note.get("objective", "Not applicable"),
        "Assessment": note.get("assessment", "Not applicable"),
        "Plan": note.get("plan", "Not applicable"),
    }

    return "\n\n".join(f"{section}:\n{text}" for section, text in sections.items())


def main() -> None:
    args = parse_args()
    load_env_file()
    validate_openai_key()

    evaluator = ClinicalNoteEvaluator(
        config=MetricConfig(
            model=args.model,
            use_llm_extraction=True,
            use_llm_matching=True,
            temperature=0.0,
        ),
        llm_client=OpenAIJudgeClient(),
    )
    print("Using OpenAI single-call LLM judge.")

    ground_truth_note = (
        load_note_file(args.ground_truth)
        if args.ground_truth
        else load_note_text(GROUND_TRUTH_NOTE_PLACEHOLDER)
    )
    generated_note = (
        load_note_file(args.generated)
        if args.generated
        else load_note_text(GENERATED_NOTE_PLACEHOLDER)
    )
    transcript = (
        Path(args.transcript).read_text(encoding="utf-8")
        if args.transcript
        else TRANSCRIPT_PLACEHOLDER
    )

    result = evaluator.evaluate(
        ground_truth_note=ground_truth_note,
        generated_note=generated_note,
        transcript=transcript,
    )

    print(f"Overall Fidelity Score: {result.overall_fidelity_score:.2f}/100")
    print(f"Summary: {result.summary}")
    print(
        "Scores: "
        f"completeness={result.scores.completeness:.1f}, "
        f"correctness={result.scores.correctness:.1f}, "
        f"supported_content={result.scores.supported_content:.1f}, "
        f"section_placement={result.scores.section_placement:.1f}, "
        f"detail_fidelity={result.scores.detail_fidelity:.1f}"
    )
    print(
        "Counts: "
        f"GT={result.counts.ground_truth_facts}, "
        f"generated={result.counts.generated_facts}, "
        f"correct={result.counts.correct}, "
        f"partial={result.counts.partial}, "
        f"missing={result.counts.missing}, "
        f"unsupported={result.counts.unsupported}, "
        f"contradictions={result.counts.contradictions}, "
        f"clinical_error_events={result.counts.clinical_error_events}"
    )
    if result.clinical_error_events:
        print("Clinical Error Profile:")
        for error in result.clinical_error_events[:5]:
            print(f"- {error.type}")
            print(f"  Severity: {error.severity}")
            print(f"  Reason: {error.reason}")
            if error.clinical_consequence:
                print(f"  Clinical consequence: {error.clinical_consequence}")
        if len(result.clinical_error_events) > 5:
            print(f"- ... {len(result.clinical_error_events) - 5} more")

    unsupported_facts = [
        extra for extra in result.unsupported_facts if extra.classification == "UNSUPPORTED"
    ]
    supported_additions = [
        extra
        for extra in result.unsupported_facts
        if extra.classification == "SUPPORTED_BUT_ABSENT_FROM_GT"
    ]

    if unsupported_facts:
        print("Unsupported Facts:")
        for extra in unsupported_facts[:5]:
            print(f"- {extra.generated_fact.evidence_text}: {extra.reason}")
        if len(unsupported_facts) > 5:
            print(f"- ... {len(unsupported_facts) - 5} more")

    if supported_additions:
        print("Transcript-Supported Additions:")
        for extra in supported_additions[:5]:
            print(f"- {extra.generated_fact.evidence_text}: {extra.reason}")
        if len(supported_additions) > 5:
            print(f"- ... {len(supported_additions) - 5} more")

    if os.getenv("CNFS_FULL_JSON", "").lower() in {"1", "true", "yes"}:
        print("\nFull JSON:")
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
