# Day 7 — BM25 and Hybrid Retrieval

## Goal

Add exact keyword retrieval and combine it with vector search.

By the end of Day 7, the project can:

- Ingest local PDFs, scans, JPEG/JPG/PNG images, and handwritten notes
- Build a persistent vector index
- Build a BM25 keyword index from the same chunks
- Apply the same metadata filters to both methods
- Combine vector and BM25 rankings with Reciprocal Rank Fusion
- Compare retrieval behavior for five synthetic questions

## Architecture

```text
Question
   ├── Vector retriever ─────┐
   │                         │
   └── BM25 retriever ───────┤
                             ↓
                   Reciprocal Rank Fusion
                             ↓
                    Unique hybrid ranking
```

## `tokenizer.py`

The tokenizer preserves terms important for medical retrieval:

```text
SYN-200849
HbA1c
6.8%
92 mg/dL
follow-up
2026-07-20
COVID-19
```

Pseudo-code:

```text
normalize Unicode
lowercase
find medical-friendly tokens
remove common stop words
return stable token list
```

Exact-value preservation is important because a patient ID or dosage should
not be broken into unrelated fragments.

## `bm25_retriever.py`

The BM25 index is built from the same chunks stored by Day 5.

Pseudo-code:

```text
read IDs, documents, and metadata from collection
tokenize each document
count term frequency
count document frequency
calculate average document length

for each query:
    tokenize query
    apply metadata filter
    calculate BM25 score
    remove zero-match documents
    sort by score
    return typed passages
```

BM25 is particularly useful for:

- Patient IDs
- Medication names
- Allergies
- Dates
- Codes
- Lab values
- Rare abbreviations

## `fusion.py`

BM25 scores and vector similarity values are not directly comparable.

Do not calculate:

```text
vector score + BM25 score
```

Instead, use Reciprocal Rank Fusion:

```text
RRF score = Σ 1 / (constant + rank)
```

A chunk returned by both retrieval methods receives contributions from both
lists.

Default:

```text
fusion constant = 60
```

This is a baseline to evaluate, not a universal optimum.

## `hybrid_retriever.py`

Pseudo-code:

```text
run vector search for top 15
run BM25 search for top 15
deduplicate by chunk ID
fuse rank positions with RRF
sort by fused score
return caller's requested top-k
```

The same `SearchFilters` object is used for both methods. This prevents vector
search from filtering by patient while BM25 accidentally searches every
patient.

## Run the complete offline demo

```bash
$ python scripts/run_hybrid_search.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reset
```

Filter to one patient:

```bash
$ python scripts/run_hybrid_search.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --patient-id SYN-200849 \
  --reset
```

Use Sentence Transformers and ChromaDB:

```bash
$ pip install -r requirements-day-05-full.txt

$ python scripts/run_hybrid_search.py data/samples \
  --embedding-backend sentence-transformer \
  --storage-backend chroma \
  --reset
```

## Run all tests

```bash
$ python -m pytest \
  tests/integration/test_chroma_ingestion.py \
  tests/integration/test_vector_retrieval.py \
  tests/unit/test_tokenizer.py \
  tests/unit/test_fusion.py \
  tests/integration/test_hybrid_retrieval.py \
  -v
```

## Test coverage

### Tokenizer

- Patient IDs
- Laboratory abbreviations
- Dosages
- Dates
- Hyphenated terms
- Stop-word removal
- Unicode normalization
- Deterministic output

### Fusion

- Duplicate chunk deduplication
- Contributions from both retrieval methods
- Top-n truncation
- Hybrid metadata
- Invalid configuration rejection

### Hybrid integration

- PDF and handwritten-image ingestion
- BM25 index rebuilding
- Exact patient-ID retrieval
- Unique fused results
- Patient isolation
- Exact allergy retrieval
- No invented cardiology-referral metadata

## Synthetic questions

```text
What medication was prescribed at discharge?
What allergies are documented?
When is the follow-up appointment?
What was the primary diagnosis?
Which patient had a cardiology referral?
```

The first and fifth questions are not fully supported by the current samples.
Retrieval may return related passages, but later RAG stages must distinguish
related context from sufficient evidence.

## Acceptance criteria

- [ ] Local ingestion still supports all required formats.
- [ ] BM25 indexes every stored chunk.
- [ ] Exact patient IDs are retrievable.
- [ ] Metadata filters apply to BM25 and vector search.
- [ ] RRF deduplicates chunks.
- [ ] Hybrid results contain per-method ranks and scores.
- [ ] Cross-patient retrieval is prevented.
- [ ] Unit and integration tests pass.
- [ ] The example script runs end to end.
- [ ] The Week 1 report documents answerability limitations.

## Design questions

1. Why is BM25 useful when vector search already exists?
2. Why should raw BM25 and vector scores not be added?
3. Why does RRF use rank rather than score?
4. How does tokenization affect medication and patient-ID retrieval?
5. Why must metadata filtering happen in both retrieval paths?
6. When might BM25 outperform vector search?
7. When might vector search outperform BM25?
8. Why can hybrid retrieval still return insufficient evidence?
9. How would you tune vector and BM25 candidate counts?
10. What will reranking add on Day 8?
