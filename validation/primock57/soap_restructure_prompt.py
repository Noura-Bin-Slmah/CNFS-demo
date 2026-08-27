"""Prompt for reorganizing a free-text primock57 note into CNFS's SOAP
sections, as a preprocessing step before validation. This is a pure
reorganization pass: every clinical statement must be preserved, only
recategorized into a section. The human-derived error labels in
results.csv were extracted against note *content*, not layout, so they
remain a valid target after this transform.

The section definitions below are copied verbatim from
clinical_note_metric/prompts.py so this preprocessing step uses the same
rubric CNFS itself will apply later. Kept as a static copy (not imported)
since this is a one-off validation utility, not a production dependency.
"""

from __future__ import annotations

SOAP_RESTRUCTURE_PROMPT = """You are reorganizing a free-text clinical consultation note into a
structured SOAP note. The input note has no section headers — it is a
sequence of shorthand fragments, bullet lines, or sentences. Your only job
is to sort its existing content into the correct section below. Do not
summarize, compress, paraphrase away detail, add anything not stated in
the note, or drop any clinical statement.

Section definitions (use these, and only these):
- Problem List: only clearly stated or strongly implied active problems,
  symptoms, diagnoses, or disease entities. Exclude procedures, tests,
  medications, and administrative instructions.
- Subjective: patient-reported symptoms, history, lifestyle, medication
  adherence, concerns, and patient answers.
- Objective: measurable or observed data stated by the doctor, including
  examination, vitals, imaging, and laboratory results.
- Assessment: clinician impressions or diagnoses, only when stated or
  clearly explained.
- Plan: only actions the doctor explicitly tells the patient to do now or
  next, or explicitly orders/prescribes/schedules.

Rules:
- Every clinical statement in the input (symptom, history detail,
  examination finding, diagnosis, plan action) must appear in the output,
  in the section it belongs to, preserved close to its original wording.
- The only content you may omit is pure call/consultation logistics with
  no clinical meaning (e.g. "ID check completed", "patient is at home in
  a private location", greetings). If in doubt, keep it rather than drop
  it.
- If a statement doesn't clearly fit a section's definition, place it in
  the closest defensible section per the definitions above — do not
  invent a 6th section and do not discard it.
- A section with no content for this note gets its header followed by
  "None documented." — do not skip a header.
- Output plain text only, in this exact format, nothing else:

Problem List:
- <line>

Subjective:
<line>

Objective:
<line>

Assessment:
<line>

Plan:
<line>

Input note:
---
{note_text}
---
"""
