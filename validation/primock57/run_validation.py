"""Run CNFS against data/ground_truth.csv (doctor notes, SOAP-restructured)
paired with data/generated.csv (one AI model's notes, SOAP-restructured)
by `consultation`, and record the metric's scores alongside the
human-derived error counts, so the two can be correlated afterwards
(analyze_correlation.py). Read-only against the main package; writes only
inside this validation/primock57/ folder.

Requires build_validation_set.py -> restructure_to_soap.py ->
export_soap_pairs.py to have already produced both input files.

Usage:
    python run_validation.py --sample 15
    python run_validation.py --sample 0          # run every paired row
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from clinical_note_metric.config import MetricConfig  # noqa: E402
from clinical_note_metric.evaluator import ClinicalNoteEvaluator  # noqa: E402
from clinical_note_metric.openai_client import OpenAIJudgeClient  # noqa: E402

DEFAULT_MODEL = os.getenv("CNFS_OPENAI_MODEL", "gpt-4.1-mini")

GROUND_TRUTH_CSV = os.path.join(os.path.dirname(__file__), "data", "ground_truth.csv")
GENERATED_CSV = os.path.join(os.path.dirname(__file__), "data", "generated.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "cnfs_vs_human.csv")

FIELDNAMES = [
    "consultation",
    "model",
    "cnfs_final_score",
    "cnfs_completeness",
    "cnfs_correctness",
    "cnfs_supported_content",
    "cnfs_section_placement",
    "counts_missing",
    "counts_contradictions",
    "counts_incorrect",
    "counts_unsupported",
    "counts_clinical_error_events",
    "human_critical_incorrect",
    "human_noncritical_incorrect",
    "human_critical_omissions",
    "human_noncritical_omissions",
    "n_evaluators",
    "error",
]


def load_ground_truth() -> dict[str, str]:
    if not os.path.exists(GROUND_TRUTH_CSV):
        raise SystemExit(f"No {GROUND_TRUTH_CSV} — run export_soap_pairs.py first.")
    with open(GROUND_TRUTH_CSV, encoding="utf-8") as f:
        return {r["consultation"]: r["soap_note"] for r in csv.DictReader(f)}


def load_generated() -> list[dict]:
    if not os.path.exists(GENERATED_CSV):
        raise SystemExit(f"No {GENERATED_CSV} — run export_soap_pairs.py first.")
    with open(GENERATED_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done(path: str) -> set[tuple[str, str]]:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    done.add((r["consultation"], r["model"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=15, help="Rows to evaluate (0 = all rows in generated.csv)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    generated_rows = load_generated()

    runnable = []
    missing_gt = 0
    for r in generated_rows:
        gt_soap = ground_truth.get(r["consultation"])
        if gt_soap is None:
            missing_gt += 1
            continue
        runnable.append((r, gt_soap, r["soap_note"]))
    if missing_gt:
        print(f"skipping {missing_gt} rows with no matching ground_truth.csv consultation")

    random.Random(args.seed).shuffle(runnable)
    if args.sample:
        runnable = runnable[: args.sample]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    done = already_done(OUT_CSV)
    write_header = not os.path.exists(OUT_CSV)

    print(f"using model: {DEFAULT_MODEL}")
    evaluator = ClinicalNoteEvaluator(
        config=MetricConfig(model=DEFAULT_MODEL, temperature=0.0),
        llm_client=OpenAIJudgeClient(),
    )

    with open(OUT_CSV, "a", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        remaining = [t for t in runnable if (t[0]["consultation"], t[0]["model"]) not in done]
        print(f"{len(runnable)} selected, {len(runnable) - len(remaining)} already done, {len(remaining)} to run")

        for i, (row, gt_soap, gen_soap) in enumerate(remaining):
            key = (row["consultation"], row["model"])
            record = dict.fromkeys(FIELDNAMES, "")
            record.update(
                consultation=key[0],
                model=key[1],
                human_critical_incorrect=row["human_critical_incorrect"],
                human_noncritical_incorrect=row["human_noncritical_incorrect"],
                human_critical_omissions=row["human_critical_omissions"],
                human_noncritical_omissions=row["human_noncritical_omissions"],
                n_evaluators=row["n_evaluators"],
            )
            try:
                t0 = time.time()
                result = evaluator.evaluate(ground_truth_note=gt_soap, generated_note=gen_soap)
                dt = time.time() - t0
                record.update(
                    cnfs_final_score=result.final_score,
                    cnfs_completeness=result.scores.completeness,
                    cnfs_correctness=result.scores.correctness,
                    cnfs_supported_content=result.scores.supported_content,
                    cnfs_section_placement=result.scores.section_placement,
                    counts_missing=result.counts.missing,
                    counts_contradictions=result.counts.contradictions,
                    counts_incorrect=result.counts.incorrect,
                    counts_unsupported=result.counts.unsupported,
                    counts_clinical_error_events=result.counts.clinical_error_events,
                )
                print(f"[{i + 1}/{len(remaining)}] {key} -> {result.final_score:.1f} ({dt:.1f}s)")
            except Exception as e:  # noqa: BLE001 - keep the sweep going on a bad row
                record["error"] = str(e)
                print(f"[{i + 1}/{len(remaining)}] {key} -> ERROR: {e}")

            writer.writerow(record)
            out_f.flush()


if __name__ == "__main__":
    main()
