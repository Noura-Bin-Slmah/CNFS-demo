"""Materialize the final two validation files with actual SOAP-formatted
text (not raw primock57 text needing a runtime cache join):

- data/ground_truth.csv  — one row per consultation: the doctor-written
  note, restructured into a Problem List / Subjective / Objective /
  Assessment / Plan SOAP note.
- data/generated.csv     — one row per consultation covered by the chosen
  AI model: that model's note, restructured the same way, plus the
  human-flagged issue counts (averaged across evaluators) for that pair.

Both are joined by `consultation`. Requires build_validation_set.py and
restructure_to_soap.py to have already run.

Usage:
    python export_soap_pairs.py
"""

from __future__ import annotations

import csv
import hashlib
import os

VALIDATION_SET_CSV = os.path.join(os.path.dirname(__file__), "data", "validation_set.csv")
RESTRUCTURED_CSV = os.path.join(os.path.dirname(__file__), "data", "soap_restructured_notes.csv")
GROUND_TRUTH_CSV = os.path.join(os.path.dirname(__file__), "data", "ground_truth.csv")
GENERATED_CSV = os.path.join(os.path.dirname(__file__), "data", "generated.csv")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    if not os.path.exists(VALIDATION_SET_CSV):
        raise SystemExit(f"No {VALIDATION_SET_CSV} — run build_validation_set.py first.")
    if not os.path.exists(RESTRUCTURED_CSV):
        raise SystemExit(f"No {RESTRUCTURED_CSV} — run restructure_to_soap.py first.")

    restructured: dict[str, str] = {}
    with open(RESTRUCTURED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            restructured[r["text_hash"]] = r["restructured_text"]

    with open(VALIDATION_SET_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    gt_written: dict[str, bool] = {}
    gt_rows = []
    gen_rows = []
    missing = 0

    for r in rows:
        gt_hash = text_hash(r["ground_truth_note"].strip())
        gen_hash = text_hash(r["model_note"].strip())
        if gt_hash not in restructured or gen_hash not in restructured:
            missing += 1
            continue

        consultation = r["consultation"]
        if consultation not in gt_written:
            gt_rows.append({"consultation": consultation, "soap_note": restructured[gt_hash]})
            gt_written[consultation] = True

        gen_rows.append(
            {
                "consultation": consultation,
                "model": r["model"],
                "soap_note": restructured[gen_hash],
                "human_critical_incorrect": r["human_critical_incorrect"],
                "human_noncritical_incorrect": r["human_noncritical_incorrect"],
                "human_critical_omissions": r["human_critical_omissions"],
                "human_noncritical_omissions": r["human_noncritical_omissions"],
                "n_evaluators": r["n_evaluators"],
            }
        )

    if missing:
        print(f"skipping {missing} rows not yet restructured (run restructure_to_soap.py for them)")

    os.makedirs(os.path.dirname(GROUND_TRUTH_CSV), exist_ok=True)
    with open(GROUND_TRUTH_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["consultation", "soap_note"])
        writer.writeheader()
        writer.writerows(gt_rows)

    with open(GENERATED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "consultation",
                "model",
                "soap_note",
                "human_critical_incorrect",
                "human_noncritical_incorrect",
                "human_critical_omissions",
                "human_noncritical_omissions",
                "n_evaluators",
            ],
        )
        writer.writeheader()
        writer.writerows(gen_rows)

    print(f"wrote {len(gt_rows)} rows to {GROUND_TRUTH_CSV}")
    print(f"wrote {len(gen_rows)} rows to {GENERATED_CSV}")


if __name__ == "__main__":
    main()
