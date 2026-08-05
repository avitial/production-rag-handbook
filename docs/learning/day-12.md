# Day 12 — Retrieval and RAG Evaluation

## Goal

Measure retrieval quality and prepare answer-generation evaluation before
tuning chunking, embeddings, candidate counts, or confidence thresholds.

## Deliverables

```text
evaluation/dataset.jsonl
evaluation/metrics.py
evaluation/run_retrieval_evaluation.py
evaluation/run_ragas_evaluation.py
reports/baseline-evaluation.md
tests/unit/test_evaluation_metrics.py
docs/learning/day-12.md
```

Supporting test:

```text
tests/integration/test_retrieval_evaluation.py
```

## Evaluation layers

```text
Retrieval evaluation
    Hit Rate@k
    Precision@k
    Recall@k
    MRR
    nDCG@k

Generation evaluation
    faithfulness
    response relevance
    citation correctness
    abstention correctness

System evaluation
    latency
    failures
    patient isolation
    confidence decisions
```

## Dataset schema

Each JSONL row includes:

```json
{
  "question_id": "q-001",
  "question": "What allergies are documented for patient SYN-200849?",
  "patient_id": "SYN-200849",
  "reference_answer": "Latex is documented as an allergy.",
  "reference_keywords": ["allergies", "latex"],
  "relevant_filenames": ["SYN-200849_handwritten.png"],
  "relevant_sections": ["Allergies"],
  "answerable": true,
  "expected_decision": "accept"
}
```

Do not hide unanswerable examples. They are necessary to measure abstention.

## Metric formulas

### Hit Rate@k

```text
1 if any relevant result appears in top k, otherwise 0
```

### Precision@k

```text
relevant results in top k / k
```

### Recall@k

```text
relevant results retrieved / total known relevant results
```

### Reciprocal Rank

```text
1 / rank of first relevant result
```

### nDCG@k

Measures whether relevant items appear near the top using logarithmic rank
discounting.

## Run the offline evaluation

```bash
python evaluation/run_retrieval_evaluation.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reset
```

This command:

```text
ingests PDF and handwritten PNG
builds vector and BM25 indexes
runs hybrid retrieval
matches results against transparent weak labels
calculates metrics
writes JSON and Markdown reports
```

## Run with real embeddings and ChromaDB

```bash
pip install -r requirements-day-05-full.txt

python evaluation/run_retrieval_evaluation.py data/samples \
  --embedding-backend sentence-transformer \
  --storage-backend chroma \
  --reset
```

## Ragas preparation

Ragas is optional because full metrics can require evaluator model
configuration.

Create Ragas-shaped JSONL rows with:

```text
user_input
retrieved_contexts
response
reference
```

Then validate them offline:

```bash
python evaluation/run_ragas_evaluation.py \
  --input evaluation/output/ragas-input.jsonl \
  --prepare-only
```
```text
Prepared 8 Ragas samples.
```

For a full run, install and configure Ragas plus evaluator models, then omit
`--prepare-only`.


## Why deterministic metrics come first

They are:

- Fast
- Reproducible
- Inexpensive
- Easy to debug
- Independent of another judging model

Model-based evaluation adds useful semantic judgment but can itself vary.

## Run all synchronized tests

```bash
python -m pytest \
  tests/integration/test_chroma_ingestion.py \
  tests/integration/test_vector_retrieval.py \
  tests/unit/test_tokenizer.py \
  tests/unit/test_fusion.py \
  tests/integration/test_hybrid_retrieval.py \
  tests/integration/test_reranking.py \
  tests/unit/test_context_builder.py \
  tests/unit/test_prompt_builder.py \
  tests/integration/test_rag_generation.py \
  tests/unit/test_citation_validator.py \
  tests/unit/test_answer_validator.py \
  tests/unit/test_json_validator.py \
  tests/integration/test_validated_rag.py \
  tests/unit/test_confidence_policy.py \
  tests/integration/test_confidence_gating.py \
  tests/unit/test_evaluation_metrics.py \
  tests/integration/test_retrieval_evaluation.py \
  -v
```

## Acceptance criteria

- [ ] Dataset includes answerable and unanswerable questions.
- [ ] Metrics are unit-tested.
- [ ] PDF and handwritten image ingest during evaluation.
- [ ] Evaluation runner completes offline.
- [ ] JSON and Markdown reports are produced.
- [ ] Weak-label limitations are documented.
- [ ] Optional Ragas adapter fails clearly when dependencies are missing.
- [ ] All synchronized tests pass.

## Design questions

1. Why is MRR useful in addition to Hit Rate?
2. Why can Precision@k fall as k increases?
3. How should unanswerable questions be scored?
4. Why manually label relevant chunk IDs?
5. Why evaluate retrieval separately from generation?
6. What does Ragas faithfulness measure?
7. Why can an LLM judge introduce variability?
8. Which chunking parameter should be tuned first?
9. How do patient filters affect retrieval metrics?
10. What regression thresholds should block a release?