# CNFS validation against primock57 human-eval data

Isolated from the main project — everything here only *reads*
`clinical_note_metric/`; it writes exclusively inside this folder.

Source: https://github.com/babylonhealth/primock57/tree/main/human_eval_data
(paper: "Human Evaluation and Correlation with Automatic Metrics in
Consultation Note Generation"). Only `data/results.csv` (the human-eval
rows) and `data/README.md` (its data dictionary) were kept from the
source repo — the metric-scores/taxonomy files weren't needed for this.

## Why this isn't a direct drop-in

primock57's notes (`Evaluator Note`, `Model Note`) are free-text UK GP
shorthand with no section headers at all — nothing like the `Problem
List:/Subjective:/Objective:/Assessment:/Plan:` structure CNFS's prompt,
fact-granularity rules, and section-placement scoring were built for.
Feeding them in unmodified forces the judge to guess section placement for
every fact, which makes `section_placement` (and anything depending on
section structure) meaningless noise, not signal. So there's a
restructuring step before CNFS ever sees these notes — see below.

## Pipeline (run in this order)

1. **`build_validation_set.py`** → `data/validation_set.csv`
   Ground truth is the **doctor-written note** — the real clinical note a
   human doctor wrote for that consultation (`Model == "doctor"`) — not
   any individual evaluator's own note. There's exactly one doctor note
   per consultation (57 total), which removes the arbitrary "which
   evaluator's note is GT" choice entirely. `--model model_5` (default)
   pairs the doctor note against a single AI model's note per consultation
   — 28 rows for model_5, its broadest-coverage model (`--model all` keeps
   every AI model, 228 rows; `--model <id>` picks a different single one).
   Since CNFS's ground truth no longer depends on which evaluator reviewed
   the pair, every evaluator who reviewed it (up to 5) contributes their
   own human-flagged issue counts, **averaged** into one steadier signal
   instead of picking just one evaluator's opinion (kept as `n_evaluators`).

2. **`restructure_to_soap.py`** → `data/soap_restructured_notes.csv`
   Reorganizes each unique note referenced by `validation_set.csv` into
   CNFS's 5 SOAP sections (with a populated Problem List) via an LLM pass
   that reuses the *same* section definitions from
   `clinical_note_metric/prompts.py`. Pure reorganization — preserves
   every clinical statement, only recategorizes it; only
   call/consultation logistics (e.g. "ID check completed") may be
   dropped. The human-derived error labels from `results.csv` were
   extracted against note *content*, not layout, so they stay valid
   ground truth after this transform. Cached by text hash (not tied to
   which model is selected), so it's resumable and shared across reruns
   with a different `--model`. Also computes a fidelity_ratio
   (content-line count before vs. after) per note, flagged in the log
   when < 0.7 — check those manually; a low ratio can be a false alarm
   (e.g. several short facts merged onto one semicolon-separated line
   instead of staying on separate lines — no content lost, CNFS's own
   fact-granularity rules already split that pattern back out).

3. **`export_soap_pairs.py`** → `data/ground_truth.csv` + `data/generated.csv`
   Joins `validation_set.csv` with the restructuring cache and writes the
   two final files with the actual SOAP text materialized (no runtime
   cache lookups needed downstream):
   - `ground_truth.csv`: one row per consultation — `consultation, soap_note`.
   - `generated.csv`: one row per consultation covered by the selected
     model — `consultation, model, soap_note`, plus that pair's averaged
     human-flagged issue counts and `n_evaluators`.

4. **`run_validation.py`** → `results/cnfs_vs_human.csv`
   Joins the two files by `consultation`, runs the real
   `ClinicalNoteEvaluator` on each (ground truth, generated) pair, appends
   CNFS's scores alongside that row's human-flagged issue counts.
   Resumable.

5. **`analyze_correlation.py`**
   Reads `results/cnfs_vs_human.csv` and reports Pearson/Spearman
   correlation between CNFS's dimension scores and the human-flagged issue
   counts (pure Python — no numpy/scipy/pandas in this environment), plus
   a per-model breakdown (ranking each model by how well CNFS thinks it
   matches the real doctor-written note).

## Running it

```bash
cd validation/primock57
python build_validation_set.py                  # model_5 vs doctor, 28 rows (or --model <id>)
python restructure_to_soap.py --sample 0         # all unique notes referenced (<=56 for one model)
python export_soap_pairs.py                      # materializes ground_truth.csv + generated.csv
python run_validation.py --sample 0              # full set (or --sample N to smoke-test first)
python analyze_correlation.py
```

Both restructuring and evaluation cost one real OpenAI call per note/row
(uses `OPENAI_API_KEY` / `CNFS_OPENAI_MODEL` from the project's `.env`,
same as the live app). Always smoke-test with `--sample` before scaling up.
