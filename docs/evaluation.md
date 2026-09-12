# Evaluation

Offline quality harness: retrieval metrics, answer faithfulness (lexical or
LLM-as-a-judge), refusal rate, and A/B comparison of configurations.

## Corpus and golden dataset

- `eval/corpus/*.md` — a small coherent handbook (8 documents) used as the
  indexed knowledge base.
- `eval/golden_dataset.json` — 15 cases; each lists the corpus files that must
  be retrieved (`relevant_corpus_files`) plus `source_texts` used for
  faithfulness.

Document ids are **deterministic** (`uuid5(NAMESPACE_URL,
'eval-corpus:<name>')`), so the dataset references real ids with zero manual
bookkeeping.

## Workflow

```bash
# 1. index the corpus (deterministic ids; MinIO/Qdrant/Postgres must be up)
PYTHONPATH=src uv run python scripts/index_eval_corpus.py          # first time
PYTHONPATH=src uv run python scripts/index_eval_corpus.py --force  # re-index

# 2. run the evaluation (lexical faithfulness)
PYTHONPATH=src uv run python scripts/run_eval.py \
    --dataset eval/golden_dataset.json --json-out report.json

# 3. semantic faithfulness via LLM-as-a-judge
PYTHONPATH=src uv run python scripts/run_eval.py \
    --dataset eval/golden_dataset.json --llm-judge --json-out report-llm.json
```

## Metrics

| Metric | Meaning |
|---|---|
| `precision@k` | fraction of top-k retrieved docs that are relevant |
| `recall@k` | fraction of relevant docs found in top-k |
| `mrr` | 1 / rank of the first relevant doc |
| `hit_rate` | ≥1 relevant doc in top-k |
| `ndcg@k` | discounted gain, binary relevance |
| `faithfulness` | answer grounding: lexical n-gram overlap by default, semantic score with `--llm-judge` |
| `refusal_rate` | fraction of non-informative refusals |
| `errors` | cases whose answer function raised |

## LLM-as-a-judge

`evaluation/judge.py::LLMFaithfulnessJudge` sends question + answer + context
to the configured LLM and parses a JSON verdict
(`{"faithful": bool}` or `{"score": 0..1}`, also `YES/NO`). It recognizes
refusals as faithful and **falls back to the lexical heuristic** on any LLM
error or unparsable verdict, so the pipeline never breaks. A case study of the
lexical baseline lives in the module docstring.

## A/B comparisons

```python
from evaluation import EvalRunner

baseline = await EvalRunner(answer_fn_hybrid_off).run(cases)
variant = await EvalRunner(answer_fn_hybrid_on).run(cases)
from evaluation.runner import compare_reports
print(compare_reports(baseline, variant))
```

`scripts/run_eval.py` is the CLI form: run it twice with different settings
(`SEARCH_HYBRID`, `LLM_PROVIDER`, chunk sizes) and diff the JSON reports.
