# Day 12 Retrieval Evaluation

- Evaluated questions: 8
- Cutoff: k=5
- Mean Hit Rate@5: 0.750
- Mean Precision@5: 0.175
- Mean Recall@5: 1.000
- Mean Reciprocal Rank: 0.375
- Mean nDCG@5: 0.773

## Per-question results

| ID | Answerable | Hit | P@k | R@k | RR | nDCG |
|---|---:|---:|---:|---:|---:|---:|
| q-001 | True | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 |
| q-002 | True | 1.00 | 0.20 | 1.00 | 0.50 | 0.63 |
| q-003 | True | 1.00 | 0.20 | 1.00 | 0.25 | 0.43 |
| q-004 | True | 1.00 | 0.40 | 1.00 | 0.50 | 1.06 |
| q-005 | True | 1.00 | 0.20 | 1.00 | 0.25 | 0.43 |
| q-006 | False | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| q-007 | False | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| q-008 | True | 1.00 | 0.20 | 1.00 | 0.50 | 0.63 |

## Interpretation

This synthetic baseline uses filename and keyword weak labels. It verifies the evaluation plumbing and supports parameter comparisons, but it should be replaced or supplemented with human-labeled relevant chunk IDs before making production claims.
