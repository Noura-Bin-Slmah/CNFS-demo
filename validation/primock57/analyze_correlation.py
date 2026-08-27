"""Correlate CNFS scores (from run_validation.py's output) against the
human-derived error counts from primock57, to estimate how well CNFS tracks
human judgment. Pure-Python stats (no numpy/scipy/pandas dependency).

Usage:
    python analyze_correlation.py
"""

from __future__ import annotations

import csv
import math
import os

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results", "cnfs_vs_human.csv")


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(rank(xs), rank(ys))


def load_rows() -> list[dict]:
    if not os.path.exists(RESULTS_CSV):
        raise SystemExit(f"No results yet at {RESULTS_CSV} — run run_validation.py first.")
    with open(RESULTS_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]
    return rows


def as_float(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def human_error_score(r: dict) -> float:
    """Higher = more human-flagged problems (weight critical items 2x).
    Counts are averaged across the evaluators who reviewed this pair."""

    return (
        2 * float(r["human_critical_incorrect"])
        + float(r["human_noncritical_incorrect"])
        + 2 * float(r["human_critical_omissions"])
        + float(r["human_noncritical_omissions"])
    )


def restricted_score(r: dict) -> float:
    """final_score with supported_content/section_placement dropped and
    completeness/correctness renormalized to their current relative
    weights (0.30 / 0.40 in clinical_note_metric/config.py DEFAULT_WEIGHTS,
    i.e. 0.30/0.70 and 0.40/0.70 once the other two are excluded). This
    dataset has no human label for supported_content or section_placement,
    so it can only ever validate this restricted score, not the real
    final_score — this comparison tests whether excluding those two
    dimensions recovers correlation strength lost to final_score's dilution."""

    completeness_weight = 0.30 / 0.70
    correctness_weight = 0.40 / 0.70
    return completeness_weight * float(r["cnfs_completeness"]) + correctness_weight * float(r["cnfs_correctness"])


def report_pair(name: str, xs: list[float], ys: list[float]) -> None:
    r = pearson(xs, ys)
    rho = spearman(xs, ys)
    r_str = f"{r:+.3f}" if r is not None else "n/a"
    rho_str = f"{rho:+.3f}" if rho is not None else "n/a"
    print(f"  {name:<45} pearson r = {r_str:>7}   spearman rho = {rho_str:>7}   (n={len(xs)})")


def main() -> None:
    rows = load_rows()
    if not rows:
        print("No successful rows in results file yet.")
        return

    print(f"Loaded {len(rows)} successful evaluations from {RESULTS_CSV}\n")

    human_total = [human_error_score(r) for r in rows]
    human_incorrect = [2 * float(r["human_critical_incorrect"]) + float(r["human_noncritical_incorrect"]) for r in rows]
    human_omissions = [2 * float(r["human_critical_omissions"]) + float(r["human_noncritical_omissions"]) for r in rows]

    print("Correlation with human-flagged issues (negative r/rho is expected: more human-flagged issues should mean a lower CNFS score):")
    report_pair("cnfs_final_score vs total human issues", as_float(rows, "cnfs_final_score"), human_total)
    report_pair("cnfs_correctness vs human incorrect statements", as_float(rows, "cnfs_correctness"), human_incorrect)
    report_pair("cnfs_completeness vs human omissions", as_float(rows, "cnfs_completeness"), human_omissions)
    report_pair("counts_incorrect+contradictions vs human incorrect statements",
                [float(r["counts_incorrect"]) + float(r["counts_contradictions"]) for r in rows], human_incorrect)
    report_pair("counts_missing vs human omissions", as_float(rows, "counts_missing"), human_omissions)
    report_pair(
        "restricted score (completeness+correctness only) vs total human issues",
        [restricted_score(r) for r in rows],
        human_total,
    )
    print(
        "  (restricted score drops supported_content/section_placement, which this dataset has\n"
        "   no human label for, and renormalizes completeness/correctness to their current relative\n"
        "   weights - compare its r/rho above to cnfs_final_score's, to see how much of final_score's\n"
        "   weaker correlation is dilution from the two untested dimensions vs. something else.)"
    )

    print("\nPer-model average CNFS final_score vs average human issue count")
    print("(ground truth is the real doctor-written note, so this ranks each AI model by how well")
    print(" CNFS thinks it matches the actual clinical record):")
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, mrows in sorted(by_model.items(), key=lambda kv: -sum(as_float(kv[1], "cnfs_final_score")) / len(kv[1])):
        avg_score = sum(as_float(mrows, "cnfs_final_score")) / len(mrows)
        avg_human = sum(human_error_score(r) for r in mrows) / len(mrows)
        print(f"  {model:<10} n={len(mrows):<4} avg_cnfs={avg_score:6.1f}   avg_human_issues={avg_human:5.2f}")


if __name__ == "__main__":
    main()
