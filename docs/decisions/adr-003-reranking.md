# ADR-003: Rerank Hybrid Candidates with a Cross-Encoder


## Decision

Use hybrid retrieval to generate a wider candidate set, then rerank those
candidates with a cross-encoder. The production baseline is
`cross-encoder/ms-marco-MiniLM-L-6-v2`. A deterministic overlap reranker keeps
tests and offline examples runnable but is not quality-equivalent.

## Pipeline

```text
Vector + BM25 → RRF → top candidates → cross-encoder → final passages
```

Reranking preserves chunk ID, document ID, patient ID, filename, page, section,
original rank, and original score. Scores are relevance signals, not answer
confidence or medical certainty.

## Consequences

Benefits: better ordering, replaceable implementation, inspectable ranking
changes. Costs: additional latency, model memory, configuration, and no ability
to recover passages missed by first-stage retrieval.

## Observability

Write optional JSONL traces containing query, filters, retrieval/reranker
models, before/after ranks, scores, stage durations, provenance, and short text
excerpts. Do not log complete sensitive records by default.
