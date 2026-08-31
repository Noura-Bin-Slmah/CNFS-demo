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
Every task you are given defines its own operational rules for judgments that
could otherwise be subjective (fact boundaries, classifications, severities).
Apply those rules mechanically and identically every time you see the same
input — do not substitute your own judgment where an operational rule
already gives an answer, and do not let two runs of the same input diverge.
""".strip()

FACT_EXTRACTION_PROMPT = """
Prompt version: {prompt_version}
Extract atomic clinical facts from a single {note_role} medical note for
Clinical Note Fidelity Score (CNFS) evaluation. Extract facts from this note
only — it will be compared against another note in a later step, so do not
guess at, reference, or assume the content of any other note here.

Fact-granularity policy — this is the only test for where a fact boundary
goes; do not apply any other splitting or merging logic than what is given
here:
A boundary exists only where two pieces of information could receive
DIFFERENT fidelity classifications when this note is later compared
against another note — that is, one could turn out CORRECT while the
other independently turns out MISSING, PARTIAL, INCORRECT, or
CONTRADICTION. If two pieces of information must always be judged
together as one clinical assertion, they are one fact, never two, no
matter how many describable details they contain.

Apply that test through these rules, in order, and no others:
1. Measurements and labs: each independently reported measurement or lab
   value is its own fact — "BP 136/86, glucose 6.3, hemoglobin 14" is 3
   facts. Do not split one measurement into a name fact, a value fact,
   and a units fact — "Blood pressure 136/86 mmHg" is 1 fact, not 2 or 3.
   Worked example: "Labs show hemoglobin 14, normal kidney function, and
   adequate vitamin D and B12 levels" is 4 facts — hemoglobin 14, normal
   kidney function, vitamin D adequate, B12 adequate — because each is an
   independently reportable lab value, even though they are joined into
   one sentence by commas and "and".
2. Medication instructions: keep the drug, action (start/continue/stop/
   reduce/increase), dose, frequency, route, and timing together as ONE
   fact when they describe a single medication action — these are
   qualifiers of that one action, not separate facts, so a later mismatch
   in any of them is scored as PARTIAL/INCORRECT/CONTRADICTION on that one
   fact rather than producing extra facts. "Reduce atorvastatin from 40 mg
   to 10 mg daily" is 1 fact. Two separately actionable medication
   instructions are 2 facts, one per action — "continue metformin and
   reduce atorvastatin" is 2 facts.
3. Diagnosis or problem plus its status: the diagnosis/problem and its
   stated status are ONE fact. "Diabetes is controlled" is 1 fact, never
   split into a "diabetes" fact and a "controlled" fact.
4. Symptoms and their qualifiers: keep laterality, severity, location,
   duration, temporality, and directly descriptive qualifiers attached to
   the symptom they modify, as ONE fact — "persistent left foot pain
   limiting movement" is 1 fact, not separate facts for the pain, the
   laterality, the duration, and the functional limitation. A dropped or
   changed qualifier is scored as PARTIAL, INCORRECT, or CONTRADICTION on
   that one fact later, not as an extra fact now.
5. Independent findings or actions in one sentence: split only when the
   sentence names findings or actions that could truly be judged
   independently of each other. "Patient has foot pain and denies
   dizziness" is 2 facts, because one could be captured correctly while
   the other is independently missing or contradicted.
6. Plans: one plan action, together with its clinically defining target
   and qualifiers, is one fact — "follow up in three months" is 1 fact.
   Two independently actionable plan items are two facts — "continue
   metformin and reduce atorvastatin" is 2 facts.
7. Negation: keep a negation attached to the concept it negates — "no
   anemia" is 1 fact; never extract "anemia" as a fact separate from its
   negation.
8. Source and certainty: speaker attribution ("patient reports...") and
   hedging/uncertainty ("possible", "likely") stay part of the fact they
   qualify, never a separate fact on their own — they still affect
   classification later if the other note changes them.
9. Repeated readings over time: a sequence of distinct timestamped values
   for the same measurement (for example, two glucose readings from
   different occasions) is one fact per reading, because each reading is
   independently confirmable — "notes recent readings of 6.8 then 6.3" is
   2 facts.
10. Each Problem List line is exactly one fact.

Section independence — apply after rules 1-10, before assigning any id:
extract each section's facts using only that section's own text, and treat
every section as its own independent inventory. Do not merge, omit, or
reassign a fact because the same or similar content also appears in a
different section of this note — a lab value mentioned narratively in
Subjective AND restated structurally in Objective is genuinely two facts
here, one per section, because each section is graded on its own later.
The only merging that still applies is within a single section: rules
1-10 already prevent over-fragmenting a single assertion inside one
section (for example, "diabetes is controlled" stays one fact, per rule
3 — that was never two facts to begin with, regardless of section).

Section definitions (use these, and only these, to decide which section a
fact belongs to):
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

Silent extraction procedure — follow this exact sequence for this note;
do not skip a step, and do not output anything from these steps except the
final fact list:
1. Read the complete note.
2. Go section by section. For each section, list its candidate clinical
   assertions using only that section's own text.
3. For each candidate, apply the granularity policy and rules 1-10 above:
   merge qualifiers into their parent assertion, split genuinely
   independent findings or actions. Do this independently per section —
   do not check other sections for overlap.
4. Re-read each section and confirm every clinically meaningful assertion
   stated in it is represented exactly once, using only that section's
   own text.
5. Check your draft against this list, and silently fix anything that
   fails, before finalizing:
   - No medication instruction was fragmented into separate drug/action/
     dose/frequency/route facts.
   - No diagnosis-status assertion was split into a diagnosis fact and a
     status fact.
   - No symptom was separated from its laterality, severity, location,
     duration, or descriptive qualifiers.
   - No single measurement was split into a value fact and a unit fact.
   - Independent labs or findings were not merged into one fact.
   - No fact was merged, dropped, or reassigned because similar content
     appears in a different section.
   - Extracting this note again from scratch with these same rules would
     produce the same fact boundaries and the same count.
6. Only after all of the above, assign sequential ids.
Do not output the results of steps 1-5 — return only the final JSON.

Return short evidence_text only; do not expose private chain-of-thought.
Keep each fact object to exactly the 4 fields shown below (id, section,
concept, evidence_text). Do not add status, value, negation, temporality,
laterality, dose, route, or any other extra fields — the output must stay
compact. Every fact must have a unique id starting with "{id_prefix}_",
using 4-digit zero-padded sequence numbers (for example "{id_prefix}_0001").

Return JSON only in this exact shape:
{{
  "facts": [
    {{
      "id": "{id_prefix}_0001",
      "section": "Subjective",
      "concept": "...",
      "evidence_text": "..."
    }}
  ]
}}

{note_role_title} note:
{note_text}
""".strip()

MATCH_AND_CLASSIFY_PROMPT = """
Prompt version: {prompt_version}
Match already-extracted clinical facts from a ground-truth medical note
against already-extracted clinical facts from a generated medical note, for
Clinical Note Fidelity Score (CNFS). The full text of both notes is given
below for context only. The two fact lists at the end of this prompt are
the complete, authoritative inventory of clinical facts in each note — do
not add, remove, split, or merge facts, and do not introduce any fact that
is not present in one of the two lists.

Do the full clinical judging work in this single response:

1. If a transcript is provided, use it only to decide whether extra
   generated facts are SUPPORTED_BUT_ABSENT_FROM_GT instead of UNSUPPORTED.
2. Match each ground-truth fact to a generated fact semantically, by
   clinical content, searching only among generated facts in the SAME
   section as that ground-truth fact — both fact lists were already
   assigned their section during extraction, and that assignment is
   authoritative here; do not search other sections and do not reassign a
   fact's section. If no generated fact in that same section documents
   this ground-truth fact's content, it is MISSING for this section, even
   if matching content happens to exist in a different section of the
   generated note — that possibility is handled separately, in the
   section-placement reconciliation step below, and never changes this
   fact's classification.
3. Classify every ground-truth fact as CORRECT, PARTIAL, INCORRECT,
   CONTRADICTION, or MISSING.
4. Classify every generated fact that is not matched to a ground-truth
   fact in its own section as SUPPORTED_BUT_ABSENT_FROM_GT or UNSUPPORTED.
5. Identify clinically meaningful documentation fidelity error events, if
   any, and assign categorical severity based on the clinical consequence
   of the specific event.
6. Run the section-placement reconciliation pass below over the MISSING
   ground-truth facts produced above, searching all generated facts.

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
- MISSING: no generated fact in this ground-truth fact's own section
  documents its content (see the matching instructions above — matching
  is scoped to the same section; a match elsewhere is reported separately
  by the reconciliation pass, but does not change this classification).
Apply this decision order every time, top to bottom — stop at the first
step that applies, and use the same order on every run:
1. Is there a semantic match for this fact among the generated facts in
   its own section (see the matching instructions above)? If no → MISSING.
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

Clinical error event creation threshold — apply the same test every time:
- Always create an event for a CONTRADICTION on an active problem, plan,
  medication, or status. A direct reversal is always clinically
  meaningful — this is never skipped.
- For MISSING or PARTIAL facts, create an event only if losing that
  specific detail could plausibly change what a clinician does next or
  is safety-relevant. A ground-truth fact that is a narrative restatement
  of content the generated note documents properly elsewhere in its own
  matching section (for example, a routine lab value) ordinarily has no
  such consequence — apply this threshold plainly rather than assuming
  every MISSING fact needs an event.
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

Section-placement reconciliation (section_placement_issues) — this is a
purely informational report, run after steps 1-5 above, over their
results. It never changes any classification, count, or score already
produced; it only explains, for the record, where content that looks
MISSING in its own section actually turned up.

Canonical section definitions — use these, and only these, only for this
reconciliation step, to decide the one canonical section a MISSING fact's
content belongs to by content type. Do not use them anywhere else: they
never re-derive, override, or judge any fact's actual assigned section
(every fact's section stays exactly what extraction gave it), and they
never affect classification, matching, or the score.
- Problem List: active problems, symptoms, diagnoses, or disease entities.
- Subjective: patient-reported symptoms, history, lifestyle, medication
  adherence, concerns, and patient answers.
- Objective: measurable or observed data, including examination, vitals,
  imaging, and laboratory results.
- Assessment: clinician impressions or diagnoses.
- Plan: actions the doctor tells the patient to do, or orders/prescribes/
  schedules.

For each ground-truth fact classified MISSING, follow this exact two-step
search order — check step 1 first, and only do step 2 if step 1 finds
nothing:
1. Canonical-section check (fast path): using the definitions above,
   decide the one canonical section for this fact's content type. Search
   only the generated facts already in that canonical section (including
   one already matched to a different ground-truth fact — do not skip a
   generated fact just because it is spoken for elsewhere) for a semantic
   match — the same clinical action or finding. If found, report a
   section_placement_issues entry and stop; do not also run step 2.
   Worked example: a ground-truth fact says "Labs show hemoglobin 14,"
   assigned to Subjective, and is MISSING there because the generated
   note's Subjective section never restates it narratively. A lab value's
   canonical section is Objective. The generated note's Objective section
   has "Hemoglobin is 14 g/dL," already matched as CORRECT against a
   separate ground-truth Objective fact for the same value. Report the
   entry anyway, explaining the ground truth also expected this in
   Subjective, and it is a lab value (canonically Objective) which the
   generated note does document there.
2. Fallback scan (only if step 1 found nothing): search every remaining
   generated fact in every other section — including ones already
   matched elsewhere, not only ones classified UNSUPPORTED — for the
   same semantic match.
   Worked example: a ground-truth fact says "Continue metformin 500mg
   twice daily," assigned to Plan, and is MISSING there. A medication
   instruction's canonical section is also Plan, so step 1 searches Plan
   and finds nothing (the generated note has no Plan content at all).
   Step 2 then scans the other sections and finds "continue metformin
   500mg twice daily" in Assessment, classified UNSUPPORTED there.
   Report a section_placement_issues entry explaining the ground truth
   expected this in Plan (its canonical section too) but the generated
   note placed it in Assessment instead.
In both examples, leave every classification exactly as already decided
(MISSING stays MISSING, CORRECT stays CORRECT, UNSUPPORTED stays
UNSUPPORTED) — a section_placement_issues entry only adds an explanation,
it never reclassifies anything.

Run steps 1 and 2 for every MISSING fact — do not skip the search or
assume there is no match without actually checking both steps. A lab
value, vital, or similar finding that is missing in one section while
genuinely present in another is common, and step 1's canonical check
alone should usually find it when it exists — searching thoroughly is as
important as not fabricating a result.

Whether a MISSING fact ends up with a section_placement_issues entry
depends entirely on whether steps 1-2 turn up a genuine match — report
one whenever you find one, and only skip it when you genuinely find
nothing. Do not force a pairing, and do not guess, when nothing is
there.
  Worked example (no genuine counterpart — correct outcome is no entry):
  a ground-truth fact says "No anemia noted," assigned to Objective, and
  is MISSING there. Its canonical section is also Objective (an absence
  of an abnormal lab finding), so step 1 searches Objective and finds
  nothing — the generated note never mentions anemia anywhere at all.
  Step 2 scans every other section and also finds nothing. Do not create
  a section_placement_issues entry for this fact under any circumstances
  — not paired with an unrelated fact, not with a vague or partial
  resemblance. This ground-truth fact is genuinely absent from the
  generated note; the reconciliation report says nothing about it.

Before writing any section_placement_issues entry, verify it passes all
of the following — if any one fails, delete the entry instead of writing
it:
- The reason describes a positive match you actually found — it names
  the specific generated content and where it is. A reason that itself
  says "not documented," "not found," "no match," "is missing," or
  anything else negating the pairing is proof the entry should not
  exist; delete it instead of writing it that way.
- generated_fact_id names the one generated fact that actually contains
  the matching content — never a generated fact chosen because it was
  merely available, in the right general area, or the closest thing you
  could find. If you cannot name a specific generated fact whose content
  genuinely documents this ground-truth fact, there is no entry to make.
- The section your reason names as where the content was found is the
  same section as the generated_fact_id you are reporting — never name
  one section in the explanation and report a different fact's section
  in the data.
- Only pair a MISSING fact with a generated fact when they genuinely
  describe the same clinical content — do not create an entry for a
  coincidental topical resemblance (two different findings that just
  happen to concern the same condition).
  Worked example (topically related, but NOT the same fact — do not
  pair): a ground-truth fact says "No anemia reported," and is MISSING.
  A generated fact says "Hemoglobin 14 g/dL." Hemoglobin is clinically
  relevant to anemia, but stating a lab value is not the same assertion
  as stating an absence finding — one could be true while the other is
  separately documented or omitted, and a reader cannot verify "no
  anemia" from a hemoglobin number alone. Do not create an entry pairing
  them. "No anemia reported" being genuinely absent from the generated
  note — with no separate absence statement anywhere — is not resolved
  by a hemoglobin value existing elsewhere; it simply has no entry. The
  same logic applies to any other pair that is merely evidentially or
  diagnostically related rather than a restatement of the same
  assertion (a symptom and the test that would investigate it, a
  diagnosis and one lab value among several that could support it, and
  so on).

Each ground-truth fact appears in at most one section_placement_issues
entry. A generated fact may appear in more than one entry if it
genuinely explains more than one ground-truth fact restated in different
sections.

Coverage requirements:
- Every ground-truth fact id from the list below must appear exactly once
  in fact_matches.
- Every generated fact id from the list below must appear either as
  generated_fact_id in fact_matches or as generated_fact_id in
  unsupported_facts — never in both, never in neither.
- Use generated_fact_id null only when the ground-truth fact is MISSING.
- Return short reasons only; do not expose private chain-of-thought.
- Each fact_matches and unsupported_facts reason must justify only that one
  ground-truth-to-generated fact pair, using only the evidence_text of those
  two facts. Never reference other facts, other sections, or the note's
  overall/bigger-picture status in a reason. If a mismatch is only fully
  explained by a different fact (for example, a status change documented in
  Assessment), leave that out of this reason and capture it in its own
  fact_matches entry or clinical_error_event instead.

Final consistency check — verify every item below against your own draft
answer before returning it; fix anything that fails, silently, and do not
output this checklist:
1. Every ground-truth fact id appears exactly once in fact_matches.
2. Every generated fact id appears exactly once, either as
   generated_fact_id in fact_matches or in unsupported_facts — never in
   both, never in neither.
3. No fact_matches entry has a non-null generated_fact_id while its
   classification is MISSING, and — just as important, the direction
   that actually gets missed — no fact_matches entry has a null
   generated_fact_id while its classification is anything other than
   MISSING. CORRECT, PARTIAL, INCORRECT, and CONTRADICTION all require an
   actual matched generated fact; there is no such thing as a CORRECT
   match against nothing.
4. No fact is CONTRADICTION and also CORRECT or PARTIAL for the same
   ground-truth fact.
5. Equivalent paraphrases were not penalized as PARTIAL, INCORRECT, or
   CONTRADICTION.
6. No classification was changed because of section_placement_issues —
   that report is informational only and never overrides any
   classification decided in steps 1-4, no matter what it references.
7. Every section_placement_issues entry pairs a fact actually classified
   MISSING with a generated fact that genuinely describes the same
   clinical content, in a different section — whether or not that
   generated fact is also matched elsewhere. No entry's reason contains
   words negating its own pairing ("not documented," "not found," "no
   match," "is missing") — any entry that would need such wording is
   deleted instead, not written. No MISSING fact was given an entry just
   because it is missing; most have none.
8. Qualifier loss (PARTIAL) is distinguished from wrong core content
   (INCORRECT) and from a reversed core assertion (CONTRADICTION).
9. The same underlying clinical error was not reported as more than one
   clinical_error_event.
10. No reason invents clinical information not present in either note.

Return JSON only in this exact shape:
{{
  "fact_matches": [
    {{
      "ground_truth_fact_id": "gt_0001",
      "generated_fact_id": "gen_0001 or null",
      "classification": "CORRECT|PARTIAL|INCORRECT|CONTRADICTION|MISSING",
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
  ],
  "section_placement_issues": [
    {{
      "ground_truth_fact_id": "gt_0003",
      "generated_fact_id": "gen_0005",
      "reason": "short explanation of the section mismatch"
    }}
  ]
}}

Ground-truth note (context only — the ground-truth facts list below is the
authoritative inventory of its clinical facts):
{ground_truth_note}

Generated note (context only — the generated facts list below is the
authoritative inventory of its clinical facts):
{generated_note}

Transcript, if provided:
{transcript}

Ground-truth facts (authoritative; do not add, remove, split, or merge):
{ground_truth_facts_json}

Generated facts (authoritative; do not add, remove, split, or merge):
{generated_facts_json}
""".strip()
