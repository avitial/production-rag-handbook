# Day 13 Tuning Findings

## Scope

These findings are generated from the synthetic Day 12 evaluation dataset using the offline deterministic embedding backend. They validate the experiment framework and reveal relative behavior in this small dataset; they are not production-quality semantic-search benchmarks.

## Best observed configurations

- Best MRR: **vector-only** (0.431)
- Best nDCG: **vector-only** (0.822)
- Best Precision@k: **baseline-hybrid** (0.175)
- Fastest mean query time: **bm25-only** (0.053 ms)

## Comparison

| Rank | Experiment | Mode | Chunk/Overlap | Hit | Precision | Recall | MRR | nDCG | Query ms | Chunks |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | vector-only | vector | 500/75 | 0.750 | 0.175 | 1.000 | 0.431 | 0.822 | 0.597 | 18 |
| 2 | large-chunks-hybrid | hybrid | 1200/150 | 0.750 | 0.175 | 1.000 | 0.348 | 0.760 | 0.748 | 18 |
| 3 | baseline-hybrid | hybrid | 800/150 | 0.750 | 0.175 | 1.000 | 0.348 | 0.760 | 0.767 | 18 |
| 4 | hybrid-more-candidates | hybrid | 500/75 | 0.750 | 0.175 | 1.000 | 0.348 | 0.760 | 0.795 | 18 |
| 5 | medium-chunks-hybrid | hybrid | 500/75 | 0.750 | 0.175 | 1.000 | 0.348 | 0.760 | 0.798 | 18 |
| 6 | small-chunks-hybrid | hybrid | 300/40 | 0.750 | 0.175 | 1.000 | 0.348 | 0.760 | 0.881 | 18 |
| 7 | bm25-only | bm25 | 500/75 | 0.500 | 0.125 | 0.750 | 0.312 | 0.665 | 0.053 | 18 |

## Interpretation

1. Prefer configurations that improve MRR and nDCG without a large latency increase. Hit Rate alone can hide poor ranking.
2. Smaller chunks usually increase the number of indexed chunks and can isolate exact evidence, but they can lose surrounding context.
3. Larger chunks reduce chunk count but may dilute exact evidence inside broad passages.
4. BM25 should be watched for exact IDs, medications, codes, and dates. Vector retrieval is intended to help paraphrases.
5. Hybrid retrieval is most useful when its gains exceed its added complexity and latency.

## Recommended next baseline

Use **vector-only** as the next offline baseline because it achieved the best first-relevant-result ranking on this dataset. Before adopting it for production, rerun the experiment with the real Sentence Transformer backend and manually labeled relevant chunk IDs.

## Limitations

- Only two synthetic records are evaluated.
- Relevance uses filename and keyword weak labels.
- Hash embeddings validate plumbing, not semantic model quality.
- Timing values are local development measurements.
- Ties can be common on a small dataset.

## Next experiments

- Run all configs with `sentence-transformer` and real ChromaDB.
- Add 50–100 manually reviewed questions.
- Store manually labeled relevant chunk IDs.
- Compare RRF constants such as 20, 40, 60, and 100.
- Evaluate final confidence-policy decision accuracy.
- Add OCR-noise and scanned-PDF examples.
