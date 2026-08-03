# Day 5 — Reviewed Embeddings and Persistent Vector Index

It automatically uses real Sentence Transformers and ChromaDB
when installed, while retaining dependency-free fallbacks for local testing.

## Backends

### Embeddings

```text
auto                 Sentence Transformers when loadable; hash fallback
sentence-transformer require a real Sentence Transformers model
hash                 deterministic local embeddings
```

### Storage

```text
auto   real ChromaDB when installed; local persistent fallback
chroma require real chromadb
local  bundled JSON-persistent Chroma-compatible subset
```

The hash embedding fallback is for testing and pipeline demonstrations. Use a
trained Sentence Transformer for meaningful production semantic retrieval.

## Supported sources

```text
PDF
JPEG
JPG
PNG
handwritten-note images
```

## Run fully offline

```bash
$ python scripts/ingest_samples.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reset \
  --smoke-query
```

## Run with Sentence Transformers and ChromaDB

```bash
$ pip install sentence-transformers chromadb

$ python scripts/ingest_samples.py data/samples \
  --embedding-backend sentence-transformer \
  --storage-backend chroma \
  --reset \
  --smoke-query
```

## Expected first-run counters

```text
Processed files: greater than zero
Skipped files:   zero
Failed files:    zero
Chunks indexed:  greater than zero
```

Run again without `--reset`:

```text
Processed files: zero
Skipped files:   all discovered files
```

## Test

```bash
$ python -m pytest tests/integration/test_chroma_ingestion.py -v
```

The tests cover:

- End-to-end PDF and handwritten-image ingestion
- OCR injection
- Chunking
- Embeddings
- Persistent storage
- Metadata
- Registry idempotency
- Configuration-triggered rebuild
- Five sample queries
- Persistence after reopening storage

##  Offline and full backends

### Fully offline mode

This mode requires no ChromaDB installation and no model download:
```bash
$ python scripts/ingest_samples.py \
  data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reset \
  --smoke-query
```
```text
Ingestion complete
Embedding provider: deterministic-hash-384d-v1
Storage backend:    local
Discovered files:   2
Processed files:    2
Skipped files:      0
Failed files:       0
Pages:              2
Chunks generated:   18
Chunks indexed:     18
Collection count:   18

Semantic-search smoke tests

Q: What medication was prescribed at discharge?
  1. distance=0.8995 patient=SYN-200849 page=1 | Medications Current: Metformin Past: Ibuprofen
  2. distance=0.9034 patient=SYN-200849 page=1 | Vital Signs BP: 129/91 mmHg HR: 65 bpm Temp: 97.9 F Height: 176 cm Weight: 77 kg
  3. distance=0.9091 patient=SYN-200989 page=1 | Medications Current: Metformin Past: Metformin

Q: What allergies are documented?
  1. distance=0.7818 patient=SYN-200849 page=1 | Allergies Latex
  2. distance=0.7818 patient=SYN-200989 page=1 | Allergies Shellfish
  3. distance=0.8740 patient=SYN-200849 page=1 | Medications Current: Metformin Past: Ibuprofen

Q: When is the follow-up appointment?
  1. distance=0.8995 patient=SYN-200989 page=1 | ICD-10: I10 SNOMED (synthetic placeholder): SYN-573699
  2. distance=0.9333 patient=SYN-200849 page=1 | Procedures & Results Laboratory: LDL: 92 mg/dL Imaging: MRI Lumbar: Mild degenerative changes.
  3. distance=0.9587 patient=SYN-200989 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Hypertens

Q: What was the primary diagnosis?
  1. distance=0.7778 patient=SYN-200849 page=1 | Problems Active Diagnosis: Routine exam
  2. distance=0.8740 patient=SYN-200989 page=1 | Problems Active Diagnosis: Hypertension
  3. distance=0.8889 patient=SYN-200849 page=1 | Medications Current: Metformin Past: Ibuprofen

Q: Which patient had a cardiology referral?
  1. distance=0.8504 patient=SYN-200989 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Hypertens
  2. distance=0.9091 patient=SYN-200989 page=1 | ICD-10: I10 SNOMED (synthetic placeholder): SYN-573699
  3. distance=0.9101 patient=SYN-200989 page=1 | Patient Demographics Legal Name: Quinn Kim Patient ID: SYN-200989 Date of Birth: 1977-12-28 Sex: Male Race: Other Ethnicity: Hispanic or Latino
```

It uses:
- Deterministic hash embeddings
- Bundled persistent vector store
- SQLite document registry

The hash embeddings are intended for pipeline validation and offline testing, not high-quality semantic retrieval.

### Sentence Transformers and ChromaDB mode

Install the complete dependencies:
```bash
$ pip install -r requirements-day-05-full.txt
```

Then run:
```bash
$ python scripts/ingest_samples.py \
  data/samples \
  --embedding-backend sentence-transformer \
  --storage-backend chroma \
  --reset \
  --smoke-query
```
```text
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 7365.20it/s]
/home/avitial/workspace/RAG/production-rag-handbook/app/embeddings/sentence_transformer.py:51: FutureWarning: The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.
  model.get_sentence_embedding_dimension()
Ingestion complete
Embedding provider: sentence-transformers/all-MiniLM-L6-v2
Storage backend:    chroma
Discovered files:   2
Processed files:    2
Skipped files:      0
Failed files:       0
Pages:              2
Chunks generated:   18
Chunks indexed:     18
Collection count:   18

Semantic-search smoke tests

Q: What medication was prescribed at discharge?
  1. distance=1.0410 patient=SYN-200849 page=1 | Medications Current: Metformin Past: Ibuprofen
  2. distance=1.1799 patient=SYN-200989 page=1 | Medications Current: Metformin Past: Metformin
  3. distance=1.4065 patient=SYN-200849 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Routine e

Q: What allergies are documented?
  1. distance=0.6708 patient=SYN-200849 page=1 | Allergies Latex
  2. distance=0.9004 patient=SYN-200989 page=1 | Allergies Shellfish
  3. distance=1.5232 patient=SYN-200849 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Routine e

Q: When is the follow-up appointment?
  1. distance=1.2365 patient=SYN-200989 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Hypertens
  2. distance=1.2534 patient=SYN-200849 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Routine e
  3. distance=1.4761 patient=SYN-200849 page=1 | Problems Active Diagnosis: Routine exam

Q: What was the primary diagnosis?
  1. distance=0.8361 patient=SYN-200849 page=1 | Problems Active Diagnosis: Routine exam
  2. distance=1.2048 patient=SYN-200989 page=1 | Problems Active Diagnosis: Hypertension
  3. distance=1.2630 patient=SYN-200989 page=1 | *** SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.

Q: Which patient had a cardiology referral?
  1. distance=1.1048 patient=SYN-200849 page=1 | ek SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.
  2. distance=1.1958 patient=SYN-200989 page=1 | Problems Active Diagnosis: Hypertension
  3. distance=1.2405 patient=SYN-200989 page=1 | *** SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.
```

This uses:
- sentence-transformers/all-MiniLM-L6-v2
- Real persistent ChromaDB
- SQLite document registry

### Automatic mode

The CLI also supports automatic fallback:
```bash
$ python scripts/ingest_samples.py \
  data/samples \
  --embedding-backend auto \
  --storage-backend auto \
  --smoke-query
```

```text
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 8686.83it/s]
/home/avitial/workspace/RAG/production-rag-handbook/app/embeddings/sentence_transformer.py:51: FutureWarning: The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.
  model.get_sentence_embedding_dimension()
Ingestion complete
Embedding provider: sentence-transformers/all-MiniLM-L6-v2
Storage backend:    chroma
Discovered files:   2
Processed files:    2
Skipped files:      0
Failed files:       0
Pages:              2
Chunks generated:   18
Chunks indexed:     18
Collection count:   18

Semantic-search smoke tests

Q: What medication was prescribed at discharge?
  1. distance=1.0410 patient=SYN-200849 page=1 | Medications Current: Metformin Past: Ibuprofen
  2. distance=1.1799 patient=SYN-200989 page=1 | Medications Current: Metformin Past: Metformin
  3. distance=1.4065 patient=SYN-200849 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Routine e

Q: What allergies are documented?
  1. distance=0.6708 patient=SYN-200849 page=1 | Allergies Latex
  2. distance=0.9004 patient=SYN-200989 page=1 | Allergies Shellfish
  3. distance=1.5232 patient=SYN-200849 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Routine e

Q: When is the follow-up appointment?
  1. distance=1.2365 patient=SYN-200989 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Hypertens
  2. distance=1.2534 patient=SYN-200849 page=1 | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A: Routine e
  3. distance=1.4761 patient=SYN-200849 page=1 | Problems Active Diagnosis: Routine exam

Q: What was the primary diagnosis?
  1. distance=0.8361 patient=SYN-200849 page=1 | Problems Active Diagnosis: Routine exam
  2. distance=1.2048 patient=SYN-200989 page=1 | Problems Active Diagnosis: Hypertension
  3. distance=1.2630 patient=SYN-200989 page=1 | *** SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.

Q: Which patient had a cardiology referral?
  1. distance=1.1048 patient=SYN-200849 page=1 | ek SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.
  2. distance=1.1958 patient=SYN-200989 page=1 | Problems Active Diagnosis: Hypertension
  3. distance=1.2405 patient=SYN-200989 page=1 | *** SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.
```

Behavior:
Sentence Transformers installed and loadable
    → use Sentence Transformers

Otherwise
    → use deterministic hash embeddings

ChromaDB installed
    → use ChromaDB

Otherwise
    → use bundled persistent vector storage

### Run the tests
From the project repo root:
```bash
$ python -m pytest \
  tests/integration/test_chroma_ingestion.py \
  -v
```

Expected result:
```text
6 passed
=============================================== test session starts ====================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: anyio-4.14.2
collected 6 items                                                                                                                                                         
tests/integration/test_chroma_ingestion.py::test_end_to_end_ingestion PASSED                                 [ 16%]
tests/integration/test_chroma_ingestion.py::test_metadata_and_documents_are_persisted PASSED                 [ 33%]
tests/integration/test_chroma_ingestion.py::test_second_run_is_skipped PASSED                                [ 50%]
tests/integration/test_chroma_ingestion.py::test_changed_chunking_configuration_rebuilds PASSED              [ 66%]
tests/integration/test_chroma_ingestion.py::test_all_sample_questions_return_ranked_results PASSED           [ 83%]
tests/integration/test_chroma_ingestion.py::test_persistence_survives_new_store_instance PASSED              [100%]

=============================================== 6 passed in 0.32s ====================================
```
The tests validate:
- PDF ingestion
- Handwritten PNG ingestion
- OCR injection
- Section-aware chunking
- Embedding generation
- Persistent vector storage
- Patient and page metadata
- Registry creation
- Idempotent repeated ingestion
- Re-indexing after configuration changes
- All five sample queries
- Persistence after reopening the vector store