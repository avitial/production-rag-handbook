# Day 13 experiment configurations

Each JSON file defines one reproducible retrieval experiment.

Required fields:

```text
experiment_id
description
chunk_size
chunk_overlap
retrieval_mode
top_k
vector_candidates
bm25_candidates
fusion_constant
embedding_backend
storage_backend
```

Supported retrieval modes:

```text
vector
bm25
hybrid
```

Run every configuration:

```bash
python evaluation/run_experiments.py \
  data/samples \
  --configs configs/experiments \
  --reset
```

Run one configuration:

```bash
python evaluation/run_experiments.py \
  data/samples \
  --config configs/experiments/small-chunks-hybrid.json \
  --reset
```
