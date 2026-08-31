"""Restructure primock57's free-text notes into CNFS's SOAP sections, once
per unique note text. Ground truth is now the doctor-written note, one per
consultation (<=57 unique texts); generated is each AI model's note, one
per (Consultation, Model) pair (<=228 unique texts) — at most ~285 calls
for the whole validation set, not one per row.

Caches to data/soap_restructured_notes.csv, keyed by a hash of the
original text, so reruns skip work already done. Also computes a rough
fidelity check (content-line count before vs. after) to flag any
restructure that looks like it dropped content, so those can be reviewed
before being used in the actual CNFS validation run.

Usage:
    python restructure_to_soap.py --sample 6      # smoke test
    python restructure_to_soap.py --sample 0      # full dataset (570 calls)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from soap_restructure_prompt import SOAP_RESTRUCTURE_PROMPT  # noqa: E402

DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "validation_set.csv")
CACHE_CSV = os.path.join(os.path.dirname(__file__), "data", "soap_restructured_notes.csv")

FIELDNAMES = [
    "text_hash",
    "original_len",
    "restructured_len",
    "original_content_lines",
    "restructured_content_lines",
    "fidelity_ratio",
    "original_text",
    "restructured_text",
]

REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
HEADER_RE = re.compile(r"^(Problem List|Subjective|Objective|Assessment|Plan):\s*$", re.I)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def count_content_lines(text: str) -> int:
    return len([ln for ln in text.split("\n") if ln.strip()])


def count_restructured_content_lines(text: str) -> int:
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or HEADER_RE.match(stripped):
            continue
        if stripped.lower() in ("none documented.", "none documented"):
            continue
        count += 1
    return count


def restructure(client, model: str, note_text: str) -> str:
    prompt = SOAP_RESTRUCTURE_PROMPT.format(note_text=note_text)
    kwargs = {"model": model, "input": prompt}
    if not model.startswith(REASONING_PREFIXES):
        kwargs["temperature"] = 0.0
    response = client.responses.create(**kwargs)
    return response.output_text


def load_unique_notes() -> list[str]:
    with open(DATA_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen: dict[str, str] = {}
    for r in rows:
        for col in ("ground_truth_note", "model_note"):
            text = r[col].strip()
            if text:
                seen[text_hash(text)] = text
    return list(seen.values())


def already_done() -> set[str]:
    done = set()
    if os.path.exists(CACHE_CSV):
        with open(CACHE_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r["text_hash"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=6, help="Unique notes to process (0 = all ~570)")
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("CNFS_OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")

    notes = load_unique_notes()
    done = already_done()
    remaining = [n for n in notes if text_hash(n) not in done]
    if args.sample:
        remaining = remaining[: args.sample]

    print(f"{len(notes)} unique notes total, {len(done)} already cached, {len(remaining)} to process now")

    os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
    write_header = not os.path.exists(CACHE_CSV)

    with open(CACHE_CSV, "a", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, note in enumerate(remaining):
            h = text_hash(note)
            try:
                t0 = time.time()
                restructured = restructure(client, model, note)
                dt = time.time() - t0
                orig_lines = count_content_lines(note)
                new_lines = count_restructured_content_lines(restructured)
                ratio = new_lines / orig_lines if orig_lines else 0.0
                writer.writerow(
                    {
                        "text_hash": h,
                        "original_len": len(note),
                        "restructured_len": len(restructured),
                        "original_content_lines": orig_lines,
                        "restructured_content_lines": new_lines,
                        "fidelity_ratio": f"{ratio:.2f}",
                        "original_text": note,
                        "restructured_text": restructured,
                    }
                )
                out_f.flush()
                flag = "  <-- LOW FIDELITY, review" if ratio < 0.7 else ""
                print(f"[{i + 1}/{len(remaining)}] {h} lines {orig_lines}->{new_lines} ratio={ratio:.2f} ({dt:.1f}s){flag}")
            except Exception as e:  # noqa: BLE001
                print(f"[{i + 1}/{len(remaining)}] {h} -> ERROR: {e}")


if __name__ == "__main__":
    main()
