"""Prompt templates for optional LLM-as-a-judge use."""

SYSTEM_GUARDRAILS = """
You are a clinical documentation fidelity judge. Evaluate documentation fidelity only.
Never infer undocumented findings. Do not reward plausible medical additions.
Preserve uncertainty and speaker attribution. Distinguish patient/caregiver report
from clinician observation. Distinguish planned actions from completed actions.
Distinguish previous medication use from new prescriptions. Do not assume absent
information is negative. Do not interpret "not mentioned" as "normal". Do not
invent physical examination findings, diagnoses, or treatment plans. Accept
semantic paraphrases. Penalize medically meaningful contradictions. Return valid JSON.
""".strip()

SINGLE_CALL_EVALUATION_PROMPT = """
Prompt version: {prompt_version}
Evaluate a generated medical note against a ground-truth medical note for Clinical
Note Fidelity Score (CNFS). Do the full clinical judging work in this single
response:

1. Extract atomic clinical facts from the ground-truth note.
2. Extract atomic clinical facts from the generated note.
3. If transcript is provided, use it only to decide whether extra generated facts
   are SUPPORTED_BUT_ABSENT_FROM_GT instead of UNSUPPORTED.
4. Match each ground-truth fact to generated facts semantically, by clinical
   content — search the entire generated note, every section, for each
   ground-truth fact's content before deciding it is missing. Before you
   mark a ground-truth fact MISSING, and before you add a generated fact to
   unsupported_facts, re-check the whole generated note for content that
   documents the same clinical action or finding in a different section. If
   you find it, that is the match — pair them in fact_matches instead.
   Worked example: ground-truth Plan says "Continue metformin 500mg twice
   daily." The generated note has no Plan section content, but its
   Assessment section contains "continue metformin 500mg twice daily." This
   is a match, not a miss: pair them in fact_matches with classification
   CORRECT (the content is accurate) and section_score 0.0 (wrong section).
   Do not report the ground-truth fact as MISSING while separately adding
   the same generated sentence to unsupported_facts — that reports real,
   accurate content as both missing and extraneous at once, which is wrong.
5. Classify every ground-truth fact as CORRECT, PARTIAL, INCORRECT,
   CONTRADICTION, or MISSING.
6. Classify every generated fact that is not matched to a ground-truth fact as
   SUPPORTED_BUT_ABSENT_FROM_GT or UNSUPPORTED.
7. Identify clinically meaningful documentation fidelity error events, if any,
   and assign categorical severity based on the clinical consequence of the
   specific event.

Fact granularity (apply identically to both notes, every time):
- One fact = one distinct clinical concept: one problem, one symptom, one
  measurement/lab value, one diagnosis status statement, or one plan action.
- A sentence or list line naming several distinct concepts becomes several
  facts — one per concept. Example: "Blood pressure 136/86 mmHg. Blood
  glucose 6.3. Hemoglobin 14 g/dL." is 3 facts, one per measurement.
- Sentence boundaries (periods) are not what create separate facts —
  distinct clinical concepts are. A single sentence that lists several
  findings joined by commas or "and" splits exactly the same way separate
  sentences do; do not merge a comma/and-joined list into one fact just
  because it is written as one sentence.
  Worked example: "Labs show hemoglobin 14, normal kidney function, and
  adequate vitamin D and B12 levels." is 4 facts: hemoglobin 14; normal
  kidney function; vitamin D adequate; B12 adequate. Vitamin D and B12 are
  two different lab tests joined by "and" inside the same clause — each
  gets its own fact, exactly as if they had been written as two separate
  sentences.
- A single value, even with units or a qualifier, stays one fact — do not
  split one measurement into a value fact and a units/qualifier fact.
  Example: "Blood pressure 136/86 mmHg" is 1 fact, not 2.
- Each Problem List line is exactly one fact.
- Do not create a separate fact for information that only restates or
  cross-references another fact already extracted elsewhere in the note.
- When in doubt, extract at the level of "the smallest clinical statement a
  clinician would independently confirm or deny," no finer and no coarser.

Classification definitions (fact_matches.classification) — apply the same
test every time; do not let judgment drift between runs:
- CORRECT: the generated fact preserves the full clinical meaning of the
  ground-truth fact. Paraphrasing, reordering, and different wording are
  fine as long as the core assertion, status, and direction are unchanged.
- PARTIAL: the generated fact preserves the core clinical assertion and
  direction, but drops or loosens a clinically relevant qualifier, detail,
  value, or precision (a more general term instead of a specific one, a
  dropped dose/frequency/timing, a missing severity or laterality detail).
  The core assertion itself is still correct, just less complete.
- INCORRECT: the generated fact's specific content is wrong or
  unsupported — a wrong value, wrong timing, wrong location, or an
  unsupported detail — but it does not directly reverse or negate the
  ground-truth fact's core assertion.
- CONTRADICTION: the generated fact directly reverses or negates the
  ground-truth fact's core assertion — the opposite status (present vs.
  absent, controlled vs. uncontrolled), the opposite plan (reduce a dose
  vs. continue or increase it), the opposite trend (improving vs.
  worsening), or a direct negation of something the other note affirms.
  Worked example: ground truth says "reduce statin dose from 40 mg to
  10 mg daily"; generated says "continue statin therapy." This is a direct
  reversal of the plan (reduce vs. continue) — always CONTRADICTION, never
  INCORRECT or PARTIAL, regardless of how the sentence is worded.
  A numeric or timing value that differs from the ground truth — even by
  a large margin — is not, on its own, a reversal. CONTRADICTION requires
  an actual reversal of status, plan direction, or trend as described
  above; a different number for the same kind of value (a different
  duration, dose amount, or date), with no change in direction, is
  INCORRECT, not CONTRADICTION, no matter how far apart the two values
  are or how much more urgent one sounds than the other.
  Worked example: ground truth says "follow up in three months";
  generated says "follow up in three weeks." Both notes agree a follow-up
  should happen — only the timeframe value differs, and no direction or
  plan is reversed (compare to the statin example above, where the plan
  itself flips between reducing and continuing). This is INCORRECT, not
  CONTRADICTION, even though three weeks and three months are far apart.
- MISSING: no content anywhere in the generated note, in any section,
  documents this ground-truth fact (search the whole note, every section,
  before choosing MISSING — see the matching instructions above).
Apply this decision order every time, top to bottom — stop at the first
step that applies, and use the same order on every run:
1. Is there a semantic match for this fact anywhere in the generated
   note (see the matching instructions above — search every section
   before answering)? If no → MISSING.
2. Does the matched generated fact explicitly reverse or conflict with
   the ground-truth fact's core assertion (opposite status, opposite
   plan, opposite trend, direct negation)? A different number for the
   same kind of value — duration, dose, date — is not by itself a
   reversal, even when the values are far apart; only step 3 applies to
   that case. If yes → CONTRADICTION.
3. Is the matched fact's specific content factually wrong — a wrong
   value, timing, or location — without reversing the core assertion?
   If yes → INCORRECT.
4. Is the core assertion correct, but a clinically relevant qualifier or
   detail is missing, generalized, or less precise? If yes → PARTIAL.
5. Otherwise → CORRECT.

Documentation fidelity rules:
- Never infer undocumented findings.
- Do not reward plausible medical additions.
- Weigh uncertainty, speaker/source attribution, negation, temporality,
  laterality, dose, route, frequency, duration, and planned vs completed status
  when you classify a match and write its reason — reflect these details in
  the classification and reason text itself. Do not create separate structured
  fields for them.
- Do not assume absent information is negative.
- Do not interpret "not mentioned" as "normal".
- Accept semantic paraphrases.
- Penalize medically meaningful contradictions.
- Do not call ordinary missing facts safety-critical. Missing normal results,
  minor history, or descriptive details are ordinary MISSING unless you can
  state a concrete patient-safety or management consequence.
- A contradiction is not automatically safety-critical. Only safety-critical
  contradictions need additional safety language.
- Clinical error events are not numeric penalties. Report them separately from
  the fidelity score.

Section definitions (use these, and only these, to decide which section a
fact belongs to — apply identically to the ground-truth note and the
generated note):
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
No other sections are allowed.

Section placement scoring (section_score in fact_matches):
- section_score reflects whether the generated note documented this fact in
  the section it clinically belongs in, per the section definitions above —
  it is not simply "did the generated fact end up in the same section label
  as the ground-truth fact."
- section_score is binary — only 1.0 or 0.0, never any value in between.
- section_score = 1.0: the generated fact is in the same section as the
  matched ground-truth fact, or in a different section that the definitions
  above equally support (for example, a clinician impression that directly
  justifies a plan action may defensibly sit in either Assessment or Plan).
- section_score = 0.0: the generated fact is documented in a section the
  definitions above do not support — including borderline or "nearby"
  placements. If the section is not the same one, and not equally
  supported by the definitions above, it is 0.0.
- section_score is not evaluated when classification is MISSING (there is
  no generated fact to place); use 0.0 in that case.
- classification (CORRECT/PARTIAL/INCORRECT/CONTRADICTION/MISSING) must be
  judged on content accuracy alone — whether the generated fact's meaning,
  detail, and clinical substance correctly represent the ground-truth fact.
  Never lower classification, and never write "wrong section" as a reason,
  because a fact was documented in a different section. Section placement is
  scored only through section_score. A fact with perfectly accurate content
  in the wrong section is CORRECT with a low section_score, not INCORRECT.

Clinical error event creation threshold — apply the same test every time:
- Always create an event for a CONTRADICTION on an active problem, plan,
  medication, or status. A direct reversal is always clinically
  meaningful — this is never skipped.
- For MISSING or PARTIAL facts, create an event only if losing that
  specific detail could plausibly change what a clinician does next or
  is safety-relevant.
- This rule applies broadly, to every fact, not just to the specific
  example below: if the SAME underlying clinical fact is captured
  ANYWHERE in the generated note — in any section, even if the ground
  truth stated it in more than one section, or the generated note
  dropped one of two data points for the same value trend — that
  clinical fact is captured. Do not create an event just because the
  generated note has fewer restatements, repeats, or section-duplicates
  of a fact than the ground truth has. Ask "is the clinical fact itself
  (not the number of times it is written) present anywhere in the
  generated note?" before creating a MISSING/PARTIAL event.
  Worked example: the ground truth mentions "no anemia" in both
  Subjective and Objective, and separately reports two glucose readings
  (6.8 then 6.3). If the generated note keeps "no anemia" in only one of
  those two places, and keeps only the most recent glucose reading
  (6.3) while dropping the earlier one (6.8), do not create events for
  either — the clinical facts (no anemia; current glucose 6.3) are both
  still captured, so there is no management or safety consequence. The
  same logic applies identically to any other fact restated across
  sections in the ground truth (labs, vitals, history items, and so
  on) — do not treat each one as a separate special case.
- Also do not create an event for: an incidental patient-reported habit
  or context detail with no management consequence (for example, how
  the patient tracks a value at home); or a normal-finding restatement
  that adds no new information.
- For INCORRECT facts, create an event only if the specific wrong value,
  timing, or location could affect clinical understanding or management
  — not for a trivial wording or precision difference already captured
  by a PARTIAL classification instead.
- If several mismatches or labels describe the same underlying clinical
  error, return one clinical_error_event for that group — sharing one
  clinical reason — and list the other applicable labels in
  secondary_labels. Do not create separate events for the same underlying
  issue.

Allowed clinical error event types only: MISSING_DETAIL, NEGATION_ERROR,
LATERALITY_ERROR, MEDICATION_DETAIL_OMISSION, MEDICATION_ERROR, DOSE_ERROR,
ROUTE_ERROR, ALLERGY_ERROR, UNSUPPORTED_MEDICATION, UNSUPPORTED_DIAGNOSIS,
UNSUPPORTED_OBJECTIVE_FINDING, CRITICAL_OMISSION,
SAFETY_CRITICAL_CONTRADICTION, SOURCE_CERTAINTY_TRANSFORMATION.

Event type selection — pick exactly one, the most specific type that
applies; do not let the same underlying issue get different labels
between runs:
- SAFETY_CRITICAL_CONTRADICTION is reserved for a contradiction with a
  concrete, statable immediate-danger consequence beyond the plan or
  status reversal itself (for example, a reversed allergy, a reversed
  contraindication, or a reversal that would lead to a dangerous dose or
  drug). A plan or status reversal with no additional stated danger is
  not SAFETY_CRITICAL_CONTRADICTION even though its classification is
  CONTRADICTION — use the more specific type below instead (usually
  MEDICATION_ERROR, DOSE_ERROR, or NEGATION_ERROR).
- DOSE_ERROR: the medication and the decision to treat both agree, but a
  specific dose, frequency, or numeric value is wrong (for example, both
  notes agree the dose should be reduced, but state different target
  numbers).
- MEDICATION_ERROR: the medication plan itself is reversed or wrong —
  continue vs. stop vs. reduce vs. start a medication.
  Worked example: ground truth's plan is "reduce statin dose from 40 mg
  to 10 mg daily"; generated says "continue statin therapy." The plan
  itself is reversed (reduce vs. continue), not just a wrong number, and
  there is no stated immediate-danger consequence — this is
  MEDICATION_ERROR, not SAFETY_CRITICAL_CONTRADICTION and not DOSE_ERROR.
- MEDICATION_DETAIL_OMISSION: the medication and plan direction are both
  preserved, but a qualifier is dropped (route, frequency, formulation)
  without reversing anything.
- NEGATION_ERROR: a presence/absence or positive/negative finding is
  flipped for something that is not itself a medication plan (a symptom,
  a lab result, a diagnosis status).
- MISSING_DETAIL: a supporting detail is absent per the event creation
  threshold above, and none of the more specific types apply.
- Use CRITICAL_OMISSION only for a MISSING fact whose absence itself
  could be dangerous if unnoticed, not for routine missing detail.
- Use ALLERGY_ERROR, ROUTE_ERROR, LATERALITY_ERROR, UNSUPPORTED_MEDICATION,
  UNSUPPORTED_DIAGNOSIS, UNSUPPORTED_OBJECTIVE_FINDING, and
  SOURCE_CERTAINTY_TRANSFORMATION only when the mismatch is specifically
  about that exact dimension (allergy status, route of administration,
  left/right or other laterality, an unsupported addition, or a change in
  who is asserting something / how certain they are).

Allowed clinical severity values only: NONE, LOW, MODERATE, HIGH, CRITICAL.

Severity definitions (clinical_error_events.severity) — apply the same test
every time; do not let judgment drift between runs:
- NONE: use only as a placeholder when an event has no discernible clinical
  consequence at all. This should almost never be used — if there is truly
  no consequence, do not create a clinical_error_event in the first place
  (see the rule above).

- LOW: a documentation precision issue a clinician would notice but that
  would not change what happens next — a minor descriptive detail dropped,
  a slightly less specific term used, no effect on active management.

- MODERATE: could affect a clinician's understanding of the case or a
  future decision, but does not misstate or reverse an active, current
  management action, and is not an immediate safety concern.

- HIGH: a real, plausible management error if the generated note were
  acted on as written — a reversal or misstatement of an active treatment,
  dose, or plan; a status reversal for a condition currently being managed
  (controlled vs. uncontrolled, improving vs. worsening) — but not an
  immediate, direct danger to the patient.
  Worked example: ground truth's plan is "reduce statin dose from 40 mg to
  10 mg daily"; generated note says "continue statin therapy." This is an
  active medication plan reversal a clinician or patient could act on
  incorrectly — HIGH, not MODERATE (it is a real management error, not
  just an imprecise description) and not CRITICAL (a statin dose is not an
  immediate, direct danger the way an allergy or contraindication is).
  
- CRITICAL: could directly and immediately endanger the patient if acted
  on — a missed or reversed allergy, a contraindication, a wrong-drug or
  wrong-route error, or an omitted safety-critical finding with an
  immediate consequence.
Use the same test every run: if a clinician or patient acted on the
generated note exactly as written, would it change an active treatment or
plan decision (HIGH), could it immediately endanger the patient (CRITICAL),
would it only color interpretation without changing current management
(MODERATE), or would it change nothing about care (LOW)? Two runs of the
same note pair with the same underlying issue must land on the same tier —
do not vary severity for the same kind of event without a stated reason
tied to this note pair's specific evidence_text.

Coverage requirements:
- Every ground-truth fact must have a unique id starting with gt_.
- Every generated fact must have a unique id starting with gen_.
- Every ground-truth fact ID must appear exactly once in fact_matches.
- Every generated fact ID must appear either as generated_fact_id in fact_matches
  or as generated_fact_id in unsupported_facts.
- Use generated_fact_id null only when the ground-truth fact is MISSING.
- Return short reasons only; do not expose private chain-of-thought.
- Each fact_matches and unsupported_facts reason must justify only that one
  ground-truth-to-generated fact pair, using only the evidence_text of those
  two facts. Never reference other facts, other sections, or the note's
  overall/bigger-picture status in a reason. If a mismatch is only fully
  explained by a different fact (for example, a status change documented in
  Assessment), leave that out of this reason and capture it in its own
  fact_matches entry or clinical_error_event instead.
- Keep each fact object to exactly the 4 fields shown below (id, section,
  concept, evidence_text). Do not add status, value, negation, temporality,
  laterality, dose, route, or any other extra fields — the output must stay
  compact.

Final consistency check — verify every item below against your own draft
answer before returning it; fix anything that fails, silently, and do not
output this checklist:
1. Every ground-truth fact appears exactly once in fact_matches.
2. Every generated fact appears exactly once, either as generated_fact_id
   in fact_matches or in unsupported_facts — never in both, never in
   neither.
3. No fact_matches entry has a non-null generated_fact_id while its
   classification is MISSING.
4. No fact is CONTRADICTION and also CORRECT or PARTIAL for the same
   ground-truth fact.
5. Equivalent paraphrases were not penalized as PARTIAL, INCORRECT, or
   CONTRADICTION.
6. section_score reflects placement only — no classification was lowered
   because content was documented in a different section.
7. A ground-truth fact restated in more than one section was not treated
   as missing or partial if the generated note captures that clinical
   fact anywhere.
8. Qualifier loss (PARTIAL) is distinguished from wrong core content
   (INCORRECT) and from a reversed core assertion (CONTRADICTION).
9. The same underlying clinical error was not reported as more than one
   clinical_error_event.
10. No reason or evidence_text invents clinical information not present
    in either note.

Return JSON only in this exact shape:
{{
  "ground_truth_facts": [
    {{
      "id": "gt_0001",
      "section": "Subjective",
      "concept": "...",
      "evidence_text": "..."
    }}
  ],
  "generated_facts": [
    {{
      "id": "gen_0001",
      "section": "Subjective",
      "concept": "...",
      "evidence_text": "..."
    }}
  ],
  "fact_matches": [
    {{
      "ground_truth_fact_id": "gt_0001",
      "generated_fact_id": "gen_0001 or null",
      "classification": "CORRECT|PARTIAL|INCORRECT|CONTRADICTION|MISSING",
      "section_score": 1.0,
      "reason": "short explanation"
    }}
  ],
  "unsupported_facts": [
    {{
      "generated_fact_id": "gen_0002",
      "classification": "SUPPORTED_BUT_ABSENT_FROM_GT|UNSUPPORTED",
      "reason": "short explanation"
    }}
  ],
  "clinical_error_events": [
    {{
      "type": "MEDICATION_DETAIL_OMISSION",
      "severity": "MODERATE",
      "ground_truth_fact_id": "gt_0001 or null",
      "generated_fact_id": "gen_0001 or null",
      "error_group_id": "short_stable_group_key",
      "secondary_labels": ["optional related labels"],
      "reason": "short explanation"
    }}
  ]
}}

Ground-truth note:
{ground_truth_note}

Generated note:
{generated_note}

Transcript, if provided:
{transcript}
""".strip()
