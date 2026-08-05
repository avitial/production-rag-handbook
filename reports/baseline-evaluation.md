# Baseline Evaluation — Day 12

## Purpose

This baseline evaluates the retrieval layer of the synthetic Medical Document
Assistant before parameter tuning.

## Sources

```text
SYN-200849_handwritten.png
SYN-200989.pdf
```

Both are synthetic records. The pipeline supports native PDFs, scanned PDFs,
JPEG/JPG/PNG images, and image-based handwritten notes.

## Dataset

`evaluation/dataset.jsonl` contains eight examples:

- Six answerable, patient- or document-specific questions
- Two deliberately unanswerable questions
- Reference answers
- Expected keywords
- Relevant filenames and sections
- Expected confidence-policy decisions

## Offline metrics

The dependency-free runner calculates:

```text
Hit Rate@k
Precision@k
Recall@k
Mean Reciprocal Rank
nDCG@k
```

The weak relevance labels use expected filenames plus keyword/section matches.
This makes the baseline transparent and repeatable, but not equivalent to
human-labeled relevant chunk IDs.

## Run

```bash
python evaluation/run_retrieval_evaluation.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reset
```

Outputs:

```text
evaluation/output/retrieval-results.json
evaluation/output/retrieval-results.md
```

## Interpretation guidelines

- **Hit Rate@k** answers whether at least one relevant passage was retrieved.
- **Precision@k** measures how much of the returned list is relevant.
- **Recall@k** measures whether known relevant evidence was recovered.
- **MRR** rewards placing the first relevant result near the top.
- **nDCG@k** evaluates ranking order.

For unanswerable questions, retrieval metrics alone are insufficient. The
confidence-policy decision should also be evaluated for correct abstention.

## Ragas

`evaluation/run_ragas_evaluation.py` supports an offline `--prepare-only`
mode and an optional full Ragas run. Full Ragas metrics may require evaluator
LLM and embedding configuration.

Ragas metrics should supplement, not replace, deterministic retrieval metrics.
Useful dimensions include faithfulness, response relevance, context precision,
and context recall.

## Known limitations

- Only two source documents are included.
- OCR is easy compared with diverse real handwriting.
- Weak labels may over-credit keyword matches.
- The deterministic hash embedding is for pipeline validation, not semantic
  quality benchmarking.
- No clinical claims should be made from this synthetic baseline.

## Next tuning experiments

1. Compare chunk sizes: 300, 500, 800, and 1200 characters.
2. Compare overlap: 0, 50, 100, and 150 characters.
3. Compare vector-only, BM25-only, and hybrid retrieval.
4. Compare candidate counts at 5, 10, 15, and 25.
5. Use the real Sentence Transformer backend.
6. Label relevant chunk IDs manually.
7. Measure confidence-policy decision accuracy.
