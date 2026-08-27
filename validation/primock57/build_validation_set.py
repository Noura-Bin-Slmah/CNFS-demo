"""Build a curated validation set from primock57's results.csv, sized and
shaped for CNFS specifically instead of using the raw file as-is.

Ground truth is the doctor-written note (Model == "doctor") — the single,
real clinical note a human doctor wrote for that consultation — not any
individual evaluator's own note. There's exactly one doctor note per
consultation (57 total), so this also removes the arbitrary "which
evaluator's note do we use as GT" choice entirely.

Each row here is one (Consultation, AI model) pair: doctor's note as
ground truth, that model's note as generated. Since CNFS's ground truth no
longer depends on which evaluator reviewed the pair, every evaluator who
reviewed that (Consultation, Model) pair (up to 5) contributes their own
human-flagged issue counts, averaged into one steadier signal instead of
picking just one evaluator's opinion.

Output: data/validation_set.csv — consultation, model, ground_truth_note
(doctor's note), model_note, and averaged human label counts. Both note
columns still need restructure_to_soap.py before CNFS can use them (the
raw text has no SOAP section headers).

Usage:
    python build_validation_set.py                       # model_5 vs doctor note, all covered consultations
    python build_validation_set.py --model model_3        # a different single model
    python build_validation_set.py --model all            # every AI model (228 rows)
    python build_validation_set.py --limit 100             # stratified cap per model (only with --model all)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from human_labels import parse_labels  # noqa: E402

DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "results.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "data", "validation_set.csv")

FIELDNAMES = [
    "consultation",
    "model",
    "ground_truth_note",
    "model_note",
    "human_critical_incorrect",
    "human_noncritical_incorrect",
    "human_critical_omissions",
    "human_noncritical_omissions",
    "n_evaluators",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model_5", help="Single AI model id to validate, or 'all'")
    parser.add_argument("--limit", type=int, default=0, help="Stratified cap per model (only used with --model all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(DATA_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["Evaluator Note"].strip() and r["Model Note"].strip()]
    if args.model != "all":
        rows = [r for r in rows if r["Model"] in ("doctor", args.model)]

    doctor_note: dict[str, str] = {}
    for r in rows:
        if r["Model"] == "doctor":
            doctor_note[r["Consultation"]] = r["Model Note"]
    print(f"doctor notes found for {len(doctor_note)} consultations")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["Model"] != "doctor":
            groups[(r["Consultation"], r["Model"])].append(r)

    selected = []
    skipped_no_doctor = 0
    for (consultation, model), group in groups.items():
        if consultation not in doctor_note:
            skipped_no_doctor += 1
            continue
        crit_inc, noncrit_inc, crit_omi, noncrit_omi = [], [], [], []
        for r in group:
            ci, nci = parse_labels(r["Incorrect statements"])
            co, nco = parse_labels(r["Omissions"])
            crit_inc.append(ci)
            noncrit_inc.append(nci)
            crit_omi.append(co)
            noncrit_omi.append(nco)
        n = len(group)
        selected.append(
            {
                "consultation": consultation,
                "model": model,
                "ground_truth_note": doctor_note[consultation],
                "model_note": group[0]["Model Note"],
                "human_critical_incorrect": round(sum(crit_inc) / n, 2),
                "human_noncritical_incorrect": round(sum(noncrit_inc) / n, 2),
                "human_critical_omissions": round(sum(crit_omi) / n, 2),
                "human_noncritical_omissions": round(sum(noncrit_omi) / n, 2),
                "n_evaluators": n,
            }
        )

    print(f"{len(rows)} eligible rows -> {len(selected)} (Consultation, Model) pairs with a doctor-note reference"
          f" ({skipped_no_doctor} skipped, no doctor note for that consultation)")

    if args.limit:
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in selected:
            by_model[r["model"]].append(r)
        n_models = len(by_model)
        per_model_cap = max(1, args.limit // n_models)
        rng = random.Random(args.seed)
        capped = []
        for model, group in by_model.items():
            rng.shuffle(group)
            capped.extend(group[:per_model_cap])
        selected = capped
        print(f"stratified cap: {per_model_cap} per model ({n_models} models) -> {len(selected)} rows")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in selected:
            writer.writerow(r)

    print(f"wrote {len(selected)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
