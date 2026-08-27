# Clinical Note Fidelity Score (CNFS)

CNFS evaluates a generated medical note against a ground-truth note at the atomic clinical fact level. It is designed for documentation fidelity, explainability, reproducibility, and later validation against human reviewers.

## Dimensions

Default weights:

```python
{
    "completeness": 0.30,
    "correctness": 0.30,
    "supported_content": 0.20,
    "section_placement": 0.10,
    "detail_fidelity": 0.10,
}
```

Overall fidelity score:

```text
base_score = weighted dimension score
overall_fidelity_score = base_score
final_score = overall_fidelity_score
```

Clinical errors are reported separately as a clinical error profile with categorical
severity only: NONE, LOW, MODERATE, HIGH, or CRITICAL. The LLM judge proposes the
clinical error event, severity, consequence, and evidence; Python validates the
schema and groups duplicate events. The current implementation does not subtract
clinical-error severity from the fidelity score because numeric severity impact
has not yet been validated against independent clinician ratings.

Unsupported generated facts are not penalized when they are supported by a provided transcript and absent only from the abbreviated ground truth.

## Usage

```python
from clinical_note_metric import ClinicalNoteEvaluator, MetricConfig, OpenAIJudgeClient

evaluator = ClinicalNoteEvaluator(
    config=MetricConfig(model="gpt-4.1-mini"),
    llm_client=OpenAIJudgeClient(),
)
result = evaluator.evaluate(
    ground_truth_note="Subjective: The patient reports a documented symptom for 2 days.",
    generated_note="Subjective: The patient reports the documented symptom for 2 days.",
    transcript=None,
)

print(result.final_score)
print(result.model_dump_json(indent=2))
```

## LLM Judge Integration

The package does not hard-code an LLM provider. For OpenAI, install dependencies and set your API key:

```powershell
pip install -r requirements.txt
$env:OPENAI_API_KEY = "your_api_key_here"
python run_cnfs.py
```

Optionally choose a model:

```powershell
$env:CNFS_OPENAI_MODEL = "gpt-4.1-mini"
python run_cnfs.py
```

In Python:

```python
from clinical_note_metric import ClinicalNoteEvaluator, MetricConfig, OpenAIJudgeClient

evaluator = ClinicalNoteEvaluator(
    config=MetricConfig(
        model="gpt-4.1-mini",
        temperature=0.0,
    ),
    llm_client=OpenAIJudgeClient(),
)
```

The evaluator uses one LLM call to extract clinical facts, match facts, classify missing/incorrect/unsupported content, and return structured judgment including the separate clinical error profile. Python then validates the JSON, groups duplicate clinical events, and calculates deterministic scores, counts, and section summaries.

## Semantic Generalization

CNFS requires an LLM judge. Deterministic code is still used for scoring formulas, section aggregation, counts, schema validation, and clinical error event grouping, but not as a fact extractor, matcher, or clinical severity judge. For controlled local terminology, you may provide an explicit normalization map:

```python
from clinical_note_metric import ClinicalNoteEvaluator, MetricConfig

config = MetricConfig(
    lexical_normalization={
        "local synonym": "canonical concept",
    }
)

evaluator = ClinicalNoteEvaluator(config=config, llm_client=OpenAIJudgeClient())
```
