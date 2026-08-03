# Day 6 — Metadata-Filtered Vector Retrieval

## Goal

Use the persistent Day 5 index to retrieve semantically related chunks while
restricting searches by patient, source document, page, format, and date.

## Retrieval flow

```text
PDF/JPEG/JPG/PNG/handwritten note
        ↓
Day 5 extraction, OCR, chunking, embeddings, storage
        ↓
VectorSearchRequest
        ↓
SearchFilters
        ↓
Chroma where expression
        ↓
Query embedding
        ↓
Filtered vector search
        ↓
Typed RetrievedPassage results
```

## Models

`SearchFilters` supports patient ID, document ID, document type, filename,
source format, page number, and date range. All populated filters are combined
with logical AND.

`RetrievedPassage` preserves chunk ID, document ID, filename, page, section,
patient ID, raw distance, derived similarity, and source metadata.

## Filter construction

No filters returns `None`. One filter returns a direct equality expression.
Multiple filters return `$and`. Date ranges use ISO strings and `$gte`/`$lte`.

## Vector retriever pseudo-code

```text
validate request
build metadata filter
count collection
return empty response when collection is empty
embed query with the same model used for indexing
query vector store
flatten nested backend response
align IDs, text, metadata, and distances
convert each row into RetrievedPassage
return results and diagnostics
```

For cosine distance, display similarity is `1 - distance`. Similarity is a
ranking signal, not answer confidence.

## Run offline end to end

```bash
$ python scripts/run_vector_search.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reset
```
```text
Vector retrieval

Q: What medication was prescribed at discharge?
  filter: None
  1. similarity=0.1005 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen
  2. similarity=0.0966 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Vital Signs | Vital Signs BP: 129/91 mmHg HR: 65 bpm Temp: 97.9 F Height: 176 cm Weight: 77 kg
  3. similarity=0.0909 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Medications | Medications Current: Metformin Past: Metformin

Q: What allergies are documented?
  filter: None
  1. similarity=0.2182 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Allergies | Allergies Latex
  2. similarity=0.2182 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Allergies | Allergies Shellfish
  3. similarity=0.1260 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen

Q: When is the follow-up appointment?
  filter: None
  1. similarity=0.1005 patient=SYN-200989 source=SYN-200989.pdf, page 1, section ICD-10: I10 | ICD-10: I10 SNOMED (synthetic placeholder): SYN-573699
  2. similarity=0.0667 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Procedures & Results | Procedures & Results Laboratory: LDL: 92 mg/dL Imaging: MRI Lumbar: Mild degenerative changes.
  3. similarity=0.0413 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:

Q: What was the primary diagnosis?
  filter: None
  1. similarity=0.2222 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Problems | Problems Active Diagnosis: Routine exam
  2. similarity=0.1260 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Problems | Problems Active Diagnosis: Hypertension
  3. similarity=0.1111 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen

Q: Which patient had a cardiology referral?
  filter: None
  1. similarity=0.1496 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  2. similarity=0.0909 patient=SYN-200989 source=SYN-200989.pdf, page 1, section ICD-10: I10 | ICD-10: I10 SNOMED (synthetic placeholder): SYN-573699
  3. similarity=0.0899 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Patient Demographics | Patient Demographics Legal Name: Quinn Kim Patient ID: SYN-200989 Date of Birth: 1977-12-28 Sex: Male Race: Other Ethnicity: Hispa
```

Filter to one patient:

```bash
$ python scripts/run_vector_search.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --patient-id SYN-200849 \
  --reset
```
```text
Vector retrieval

Q: What medication was prescribed at discharge?
  filter: {'patient_id': 'SYN-200849'}
  1. similarity=0.1005 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen
  2. similarity=0.0966 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Vital Signs | Vital Signs BP: 129/91 mmHg HR: 65 bpm Temp: 97.9 F Height: 176cm Weight: 77 kg
  3. similarity=0.0658 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1 | ek SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.

Q: What allergies are documented?
  filter: {'patient_id': 'SYN-200849'}
  1. similarity=0.2182 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Allergies | Allergies Latex
  2. similarity=0.1260 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen
  3. similarity=0.0605 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Vital Signs | Vital Signs BP: 129/91 mmHg HR: 65 bpm Temp: 97.9 F Height: 176cm Weight: 77 kg

Q: When is the follow-up appointment?
  filter: {'patient_id': 'SYN-200849'}
  1. similarity=0.0667 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Procedures & Results | Procedures & Results Laboratory: LDL: 92 mg/dL Imaging: MRI Lumbar: Mild degenerative changes.
  2. similarity=0.0390 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  3. similarity=0.0000 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen

Q: What was the primary diagnosis?
  filter: {'patient_id': 'SYN-200849'}
  1. similarity=0.2222 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Problems | Problems Active Diagnosis: Routine exam
  2. similarity=0.1111 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen
  3. similarity=0.0925 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section ICD-10: Z00.00 | ICD-10: Z00.00 SNOMED (synthetic placeholder): SYN-152895

Q: Which patient had a cardiology referral?
  filter: {'patient_id': 'SYN-200849'}
  1. similarity=0.0880 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Patient Demographics | Patient Demographics Legal Name: Riley Kim Patient ID:SYN-200849 Date of Birth: 1987-12-09 Sex: Female Race: Native Hawaiian Ethn
  2. similarity=0.0836 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section ICD-10: Z00.00 | ICD-10: Z00.00 SNOMED (synthetic placeholder): SYN-152895
  3. similarity=0.0706 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
```

## Run with Sentence Transformers and ChromaDB

```bash
$ pip install -r requirements.txt
$ python scripts/run_vector_search.py data/samples \
  --embedding-backend sentence-transformer \
  --storage-backend chroma \
  --reset
```
```text
Vector retrieval

Q: What medication was prescribed at discharge?
  filter: None
  1. similarity=-0.0410 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen
  2. similarity=-0.1799 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Medications | Medications Current: Metformin Past: Metformin
  3. similarity=-0.4065 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:

Q: What allergies are documented?
  filter: None
  1. similarity=0.3292 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Allergies | Allergies Latex
  2. similarity=0.0996 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Allergies | Allergies Shellfish
  3. similarity=-0.5232 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:

Q: When is the follow-up appointment?
  filter: None
  1. similarity=-0.2365 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  2. similarity=-0.2534 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  3. similarity=-0.4761 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Problems | Problems Active Diagnosis: Routine exam

Q: What was the primary diagnosis?
  filter: None
  1. similarity=0.1639 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1, section Problems | Problems Active Diagnosis: Routine exam
  2. similarity=-0.2048 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Problems | Problems Active Diagnosis: Hypertension
  3. similarity=-0.2630 patient=SYN-200989 source=SYN-200989.pdf, page 1, section *** SYNTHETIC MEDICAL RECORD *** | *** SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.

Q: Which patient had a cardiology referral?
  filter: None
  1. similarity=-0.1048 patient=SYN-200849 source=SYN-200849_handwritten.png, page 1 | ek SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.
  2. similarity=-0.1958 patient=SYN-200989 source=SYN-200989.pdf, page 1, section Problems | Problems Active Diagnosis: Hypertension
  3. similarity=-0.2405 patient=SYN-200989 source=SYN-200989.pdf, page 1, section *** SYNTHETIC MEDICAL RECORD *** | *** SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.
```

## Evaluation questions

`evaluation/questions.jsonl` includes the five requested questions plus an
additional filtered allergy question. It explicitly marks questions not fully
answerable from the supplied samples. Retrieval success does not prove that an
answer exists.

## Tests

```bash
$ python -m pytest \
  tests/integration/test_chroma_ingestion.py \
  tests/integration/test_vector_retrieval.py \
  -v
```
```text
============================================== test session starts ==============================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: anyio-4.14.2
collected 15 items                                                                                                                                                        
tests/integration/test_chroma_ingestion.py::test_end_to_end_ingestion PASSED             [  6%]
tests/integration/test_chroma_ingestion.py::test_metadata_and_documents_are_persisted PASSED       [ 13%]
tests/integration/test_chroma_ingestion.py::test_second_run_is_skipped PASSED            [ 20%]
tests/integration/test_chroma_ingestion.py::test_changed_chunking_configuration_rebuilds PASSED    [ 26%]
tests/integration/test_chroma_ingestion.py::test_all_sample_questions_return_ranked_results PASSED [ 33%]
tests/integration/test_chroma_ingestion.py::test_persistence_survives_new_store_instance PASSED  [ 40%]
tests/integration/test_vector_retrieval.py::test_filter_builder_no_filters PASSED        [ 46%]
tests/integration/test_vector_retrieval.py::test_filter_builder_combines_patient_and_page PASSED [ 53%]
tests/integration/test_vector_retrieval.py::test_filter_builder_date_range PASSED        [ 60%]
tests/integration/test_vector_retrieval.py::test_unfiltered_retrieval_returns_ranked_passages PASSED [ 66%]
tests/integration/test_vector_retrieval.py::test_patient_filter_excludes_other_patient PASSED [ 73%]
tests/integration/test_vector_retrieval.py::test_nonexistent_patient_returns_empty PASSED     [ 80%]
tests/integration/test_vector_retrieval.py::test_filename_filter PASSED                  [ 86%]
tests/integration/test_vector_retrieval.py::test_citation_label PASSED                                                                                   [ 93%]
tests/integration/test_vector_retrieval.py::test_similarity_conversion PASSED            [100%]

============================================== 15 passed in 0.71s ==============================================
```

Tests cover end-to-end ingestion, filter construction, ranked results, patient
isolation, empty matches, filename filtering, citation labels, persistence.

## Acceptance criteria

- Local PDF and image ingestion still works.
- Handwritten-note OCR still works.
- Vector queries return typed results.
- Patient filters exclude other patients.
- Missing-patient filters return no passages.
- Every result includes page-level provenance.
- The example script runs end to end.
- Day 5 and Day 6 tests pass.