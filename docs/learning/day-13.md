# Day 13 — Retrieval Tuning Experiments

## Goal

Compare chunking and retrieval configurations using the Day 12 evaluation
dataset, then choose the next baseline based on measured results.

The package continues to ingest local:

```text
PDF
scanned PDF
JPEG
JPG
PNG
handwritten-note images
```

## Deliverables

```text
configs/experiments/
evaluation/run_experiments.py
reports/retrieval-comparison.csv
reports/tuning-findings.md
docs/learning/day-13.md
```

Supporting files:

```text
tests/unit/test_experiment_runner.py
tests/integration/test_experiment_runner.py
```

## Experiment variables

Day 13 compares:

```text
chunk size
chunk overlap
retrieval mode
vector candidate count
BM25 candidate count
RRF constant
final top-k
```

## Configuration example

```json
{
  "experiment_id": "small-chunks-hybrid",
  "description": "Smaller chunks for more focused passages.",
  "chunk_size": 300,
  "chunk_overlap": 40,
  "retrieval_mode": "hybrid",
  "top_k": 5,
  "vector_candidates": 15,
  "bm25_candidates": 15,
  "fusion_constant": 60,
  "embedding_backend": "hash",
  "storage_backend": "local"
}
```

## Why isolate each experiment?

Chunk size and overlap change the actual indexed records.

Each experiment therefore uses a separate:

```text
vector directory
collection
SQLite registry
results file
```

Reusing the same index would make the comparison invalid.

## Experiment flow

```text
load config
    ↓
create isolated storage
    ↓
ingest PDF and handwritten image
    ↓
chunk using experiment parameters
    ↓
build vector and BM25 indexes
    ↓
select vector, BM25, or hybrid retrieval
    ↓
run all evaluation questions
    ↓
calculate metrics and latency
    ↓
write detail JSON
    ↓
append comparison CSV row
```

## Included configurations

```text
baseline-hybrid
small-chunks-hybrid
medium-chunks-hybrid
large-chunks-hybrid
vector-only
bm25-only
hybrid-more-candidates
```

## Run all experiments

```bash
python evaluation/run_experiments.py \
  data/samples \
  --configs configs/experiments \
  --reset
```
```text
Running baseline-hybrid from baseline-hybrid.json...
  MRR=0.348 nDCG=0.760 Hit=0.750 query=0.781 ms
Running bm25-only from bm25-only.json...
  MRR=0.312 nDCG=0.665 Hit=0.500 query=0.053 ms
Running hybrid-more-candidates from hybrid-more-candidates.json...
  MRR=0.348 nDCG=0.760 Hit=0.750 query=0.805 ms
Running large-chunks-hybrid from large-chunks-hybrid.json...
  MRR=0.348 nDCG=0.760 Hit=0.750 query=0.748 ms
Running medium-chunks-hybrid from medium-chunks-hybrid.json...
  MRR=0.348 nDCG=0.760 Hit=0.750 query=0.779 ms
Running small-chunks-hybrid from small-chunks-hybrid.json...
  MRR=0.348 nDCG=0.760 Hit=0.750 query=0.751 ms
Running vector-only from vector-only.json...
  MRR=0.431 nDCG=0.822 Hit=0.750 query=0.588 ms

Experiment comparison complete
  experiments: 7
  CSV: /home/avitial/workspace/RAG/production-rag-handbook/reports/retrieval-comparison.csv
  Findings: /home/avitial/workspace/RAG/production-rag-handbook/reports/tuning-findings.md
```

Outputs:

```text
reports/retrieval-comparison.csv
reports/tuning-findings.md
evaluation/experiment-output/<experiment>/results.json
```

## Run one experiment

```bash
python evaluation/run_experiments.py \
  data/samples \
  --config configs/experiments/small-chunks-hybrid.json \
  --reset
```

## Comparison fields

The CSV records:

```text
experiment ID
retrieval mode
chunk size and overlap
candidate settings
embedding model
Hit Rate@k
Precision@k
Recall@k
MRR
nDCG@k
ingestion latency
mean query latency
indexed chunk count
failed-file count
```

## How to choose a baseline

Prioritize:

1. MRR, because the first relevant passage should rank early.
2. nDCG, because relevant evidence should be concentrated near the top.
3. Hit Rate, because relevant evidence must be present.
4. Precision, because noisy contexts consume tokens and can mislead generation.
5. Latency and index size.

Do not choose solely by one metric.

## Interpreting chunk size

### Small chunks

Potential benefits:

```text
focused evidence
better exact-section isolation
less irrelevant text per result
```

Potential costs:

```text
more chunks
more indexing work
loss of surrounding context
```

### Large chunks

Potential benefits:

```text
more complete context
fewer chunks
less fragmentation
```

Potential costs:

```text
diluted evidence
more irrelevant text
larger generation context
```

## Retrieval-mode tradeoffs

### Vector only

Best suited to semantic similarity and paraphrases.

### BM25 only

Best suited to exact IDs, medication names, codes, dates, and rare terms.

### Hybrid

Useful when both semantic and exact-match recall matter. It adds candidate
generation and fusion complexity.

## Current limitations

The reviewed experiment uses:

```text
two synthetic documents
eight questions
weak filename/keyword relevance labels
deterministic hash embeddings
```

The results validate the tuning framework. They do not establish production
quality.

## Run synchronized tests

```bash
python -m pytest \
  tests/unit/test_evaluation_metrics.py \
  tests/integration/test_retrieval_evaluation.py \
  tests/unit/test_experiment_runner_unit.py \
  tests/integration/test_experiment_runner.py \
  -v
```
```text
================================================== test session starts ==================================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: langsmith-0.10.15, anyio-4.14.2
collected 16 items                                                                                                                                                        

tests/unit/test_evaluation_metrics.py::test_hit_rate_at_k PASSED                                                                                                    [  6%]
tests/unit/test_evaluation_metrics.py::test_precision_at_k_uses_requested_denominator PASSED                                                                        [ 12%]
tests/unit/test_evaluation_metrics.py::test_recall_at_k PASSED                                                                                                      [ 18%]
tests/unit/test_evaluation_metrics.py::test_unanswerable_recall_rewards_no_relevant_results PASSED                                                                  [ 25%]
tests/unit/test_evaluation_metrics.py::test_reciprocal_rank PASSED                                                                                                  [ 31%]
tests/unit/test_evaluation_metrics.py::test_ndcg_is_one_for_ideal_ranking PASSED                                                                                    [ 37%]
tests/unit/test_evaluation_metrics.py::test_ndcg_penalizes_late_relevance PASSED                                                                                    [ 43%]
tests/unit/test_evaluation_metrics.py::test_decision_accuracy PASSED                                                                                                [ 50%]
tests/unit/test_evaluation_metrics.py::test_summary_means_rows PASSED                                                                                               [ 56%]
tests/unit/test_evaluation_metrics.py::test_invalid_k_is_rejected PASSED                                                                                            [ 62%]
tests/integration/test_retrieval_evaluation.py::test_retrieval_evaluation_script_runs PASSED                                                                        [ 68%]
tests/unit/test_experiment_runner_unit.py::test_valid_config_loads PASSED                                                                                           [ 75%]
tests/unit/test_experiment_runner_unit.py::test_invalid_overlap_is_rejected PASSED                                                                                  [ 81%]
tests/unit/test_experiment_runner_unit.py::test_invalid_retrieval_mode_is_rejected PASSED                                                                           [ 87%]
tests/unit/test_experiment_runner_unit.py::test_findings_identify_best_experiment PASSED                                                                            [ 93%]
tests/integration/test_experiment_runner.py::test_experiment_runner_generates_csv_and_findings PASSED                                                               [100%]

================================================== 16 passed in 2.53s ==================================================
```

The complete package also retains all Day 5–12 tests.

## Acceptance criteria

- [ ] Multiple experiment configurations exist.
- [ ] Every experiment uses isolated storage.
- [ ] All required local formats still ingest.
- [ ] Vector, BM25, and hybrid modes run.
- [ ] Comparison CSV is generated.
- [ ] Findings are generated from actual results.
- [ ] Per-experiment detail JSON is preserved.
- [ ] Unit and integration tests pass.
- [ ] The full example works end to end.

## Design questions

1. Why must each chunking experiment rebuild the index?
2. Why use MRR and nDCG together?
3. Why can Hit Rate remain constant while MRR changes?
4. Why might smaller chunks improve precision?
5. Why might larger chunks improve answer generation?
6. When should BM25 outperform vector retrieval?
7. Why should timing results be treated cautiously?
8. What makes an experiment reproducible?
9. Why change one variable at a time?
10. Which configuration should become the next baseline?