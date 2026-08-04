# Day 8 — Cross-Encoder Reranking and Retrieval Tracing

## Goal

Rerank hybrid candidates before sending a small context set to an LLM. The
package continues to ingest native/scanned PDFs, JPEG/JPG/PNG images, and
handwritten-note images.

## Deliverables

- `app/reranking/base.py`: request/response models, reranker interface, shared sorting.
- `app/reranking/cross_encoder.py`: learned CrossEncoder and deterministic fallback.
- `app/observability/retrieval_trace.py`: JSONL traces with before/after ranks.
- `tests/integration/test_reranking.py`: end-to-end ingestion, retrieval, reranking, tracing.
- `docs/decisions/adr-003-reranking.md`: architecture decision and tradeoffs.

## Offline run

```bash
$ python scripts/run_reranked_search.py data/samples   --embedding-backend hash   --storage-backend local   --reranker-backend deterministic   --reset
```
```text
176 cm Weight: 77 kg
  3. rerank=0.0000 original_rank=3 patient=SYN-200989 SYN-200989.pdf, page 1, section Medications | Medications Current: Metformin Past: Metformin

Q: What allergies are documented?
  1. rerank=0.8500 original_rank=1 patient=SYN-200989 SYN-200989.pdf, page 1, section Allergies | Allergies Shellfish
  2. rerank=0.8500 original_rank=2 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Allergies | Allergies Latex
  3. rerank=0.0000 original_rank=3 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen

Q: When is the follow-up appointment?
  1. rerank=0.8000 original_rank=1 patient=SYN-200989 SYN-200989.pdf, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  2. rerank=0.8000 original_rank=2 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routinefollow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  3. rerank=0.0000 original_rank=3 patient=SYN-200989 SYN-200989.pdf, page 1, section ICD-10: I10 | ICD-10: I10 SNOMED (synthetic placeholder): SYN-573699

Q: What was the primary diagnosis?
  1. rerank=0.8000 original_rank=1 patient=SYN-200989 SYN-200989.pdf, page 1, section Problems | Problems Active Diagnosis: Hypertension
  2. rerank=0.8000 original_rank=2 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Problems | Problems Active Diagnosis: Routine exam
  3. rerank=0.0000 original_rank=3 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen

Q: Which patient had a cardiology referral?
  1. rerank=0.3333 original_rank=2 patient=SYN-200989 SYN-200989.pdf, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  2. rerank=0.3333 original_rank=6 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routinefollow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  3. rerank=0.1667 original_rank=1 patient=SYN-200989 SYN-200989.pdf, page 1, section Patient Demographics | Patient Demographics Legal Name: Quinn Kim Patient ID: SYN-200989 Date of Birth: 1977-12-28 Sex: Male Race: Other Ethnicity: Hispa

Trace file: /home/avitial/workspace/RAG/production-rag-handbook/logs/retrieval-traces.jsonl
```
## Learned-model run

```bash
$ pip install -r requirements.txt
$ python scripts/run_reranked_search.py data/samples   --embedding-backend sentence-transformer   --storage-backend chroma   --reranker-backend cross-encoder   --reset
```
```text
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 8399.70it/s]
/home/avitial/workspace/RAG/production-rag-handbook/app/embeddings/sentence_transformer.py:58: FutureWarning: The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.
  model.get_sentence_embedding_dimension()
Ingestion
  processed: 2
  skipped:   0
  failed:    0
  indexed:   18
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 8532.11it/s]
Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

Q: What medication was prescribed at discharge?
  1. rerank=-5.5654 original_rank=1 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Medications | Medications Current: Metformin Past: Ibuprofen
  2. rerank=-6.6691 original_rank=2 patient=SYN-200989 SYN-200989.pdf, page 1, section Medications | Medications Current: Metformin Past: Metformin
  3. rerank=-8.8819 original_rank=8 patient=SYN-200989 SYN-200989.pdf, page 1, section ICD-10: I10 | ICD-10: I10 SNOMED (synthetic placeholder): SYN-573699

Q: What allergies are documented?
  1. rerank=-3.0972 original_rank=2 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Allergies | Allergies Latex
  2. rerank=-3.9293 original_rank=1 patient=SYN-200989 SYN-200989.pdf, page 1, section Allergies | Allergies Shellfish
  3. rerank=-11.1222 original_rank=3 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:

Q: When is the follow-up appointment?
  1. rerank=-1.7816 original_rank=2 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  2. rerank=-2.4772 original_rank=1 patient=SYN-200989 SYN-200989.pdf, page 1, section Clinical Notes (SOAP) | Clinical Notes (SOAP) S: Patient reports routine follow-up with mild intermittent symptoms. O: Appears well. No acute distress. A:
  3. rerank=-11.2577 original_rank=6 patient=SYN-200989 SYN-200989.pdf, page 1, section Patient Demographics | Patient Demographics Legal Name: Quinn Kim Patient ID: SYN-200989 Date of Birth: 1977-12-28 Sex: Male Race: Other Ethnicity: Hispa

Q: What was the primary diagnosis?
  1. rerank=-5.7284 original_rank=1 patient=SYN-200989 SYN-200989.pdf, page 1, section Problems | Problems Active Diagnosis: Hypertension
  2. rerank=-7.8317 original_rank=2 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Problems | Problems Active Diagnosis: Routine exam
  3. rerank=-10.7409 original_rank=9 patient=SYN-200989 SYN-200989.pdf, page 1, section Procedures & Results | Procedures & Results Laboratory: HbA1c: 6.8% Imaging: Knee X-ray: Mild osteoarthritis.

Q: Which patient had a cardiology referral?
  1. rerank=-10.8410 original_rank=2 patient=SYN-200989 SYN-200989.pdf, page 1, section *** SYNTHETIC MEDICAL RECORD *** | *** SYNTHETIC MEDICAL RECORD *** NOTICE: Entirely fictional. No real patient information.
  2. rerank=-10.9721 original_rank=7 patient=SYN-200989 SYN-200989.pdf, page 1, section Problems | Problems Active Diagnosis: Hypertension
  3. rerank=-10.9773 original_rank=8 patient=SYN-200849 SYN-200849_handwritten.png, page 1, section Problems | Problems Active Diagnosis: Routine exam

Trace file: /home/avitial/workspace/RAG/production-rag-handbook/logs/retrieval-traces.jsonl
```

## Pseudo-code

```text
retrieve 10–30 hybrid candidates
score each (query, passage) pair
sort descending by reranker score
keep top 3–8
preserve page and patient provenance
write a JSONL trace
```

A reranker only reorders candidates. It cannot recover a missed passage and it
cannot prove answerability. The cardiology-referral question remains
unsupported by the supplied samples.

## Test command

```bash
$ python -m pytest   tests/integration/test_chroma_ingestion.py   tests/integration/test_vector_retrieval.py   tests/unit/test_tokenizer.py   tests/unit/test_fusion.py   tests/integration/test_hybrid_retrieval.py   tests/integration/test_reranking.py -v
```

```text
========================================== test session starts ==========================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: anyio-4.14.2
collected 36 items                                                                                                                                                        

tests/integration/test_chroma_ingestion.py::test_end_to_end_ingestion PASSED            [  2%]
tests/integration/test_chroma_ingestion.py::test_metadata_and_documents_are_persisted PASSED       [  5%]
tests/integration/test_chroma_ingestion.py::test_second_run_is_skipped PASSED           [  8%]
tests/integration/test_chroma_ingestion.py::test_changed_chunking_configuration_rebuilds PASSED [ 11%]
tests/integration/test_chroma_ingestion.py::test_all_sample_questions_return_ranked_results PASSED [ 13%]
tests/integration/test_chroma_ingestion.py::test_persistence_survives_new_store_instance PASSED    [ 16%]
tests/integration/test_vector_retrieval.py::test_filter_builder_no_filters PASSED       [ 19%]
tests/integration/test_vector_retrieval.py::test_filter_builder_combines_patient_and_page PASSED   [ 22%]
tests/integration/test_vector_retrieval.py::test_filter_builder_date_range PASSED                  [ 25%]
tests/integration/test_vector_retrieval.py::test_unfiltered_retrieval_returns_ranked_passages PASSED  [ 27%]
tests/integration/test_vector_retrieval.py::test_patient_filter_excludes_other_patient PASSED       [ 30%]
tests/integration/test_vector_retrieval.py::test_nonexistent_patient_returns_empty PASSED           [ 33%]
tests/integration/test_vector_retrieval.py::test_filename_filter PASSED                             [ 36%]
tests/integration/test_vector_retrieval.py::test_citation_label PASSED                              [ 38%]
tests/integration/test_vector_retrieval.py::test_similarity_conversion PASSED                       [ 41%]
tests/unit/test_tokenizer.py::test_preserves_medical_identifiers_and_values PASSED                  [ 44%]
tests/unit/test_tokenizer.py::test_removes_common_stop_words_by_default PASSED                      [ 47%]
tests/unit/test_tokenizer.py::test_stop_word_removal_can_be_disabled PASSED                         [ 50%]
tests/unit/test_tokenizer.py::test_unicode_is_normalized PASSED                                     [ 52%]
tests/unit/test_tokenizer.py::test_tokenization_is_deterministic PASSED                             [ 55%]
tests/unit/test_fusion.py::test_duplicate_chunk_is_fused_once PASSED                                [ 58%]
tests/unit/test_fusion.py::test_top_n_limits_results PASSED                                         [ 61%]
tests/unit/test_fusion.py::test_ranked_conversion_marks_hybrid_metadata PASSED                      [ 63%]
tests/unit/test_fusion.py::test_invalid_fusion_constant_is_rejected PASSED                          [ 66%]
tests/integration/test_hybrid_retrieval.py::test_bm25_rebuilds_from_ingested_chunks PASSED          [ 69%]
tests/integration/test_hybrid_retrieval.py::test_bm25_finds_exact_patient_identifier PASSED         [ 72%]
tests/integration/test_hybrid_retrieval.py::test_hybrid_retrieval_returns_unique_ranked_chunks PASSED [ 75%]
tests/integration/test_hybrid_retrieval.py::test_hybrid_patient_filter_prevents_cross_patient_results PASSED  [ 77%]
tests/integration/test_hybrid_retrieval.py::test_keyword_and_hybrid_find_exact_allergy PASSED       [ 80%]
tests/integration/test_hybrid_retrieval.py::test_unanswerable_cardiology_query_does_not_create_false_metadata PASSED  [ 83%]
tests/integration/test_reranking.py::test_reranker_promotes_exact_allergy_passage PASSED           [ 86%]
tests/integration/test_reranking.py::test_provenance_survives PASSED                               [ 88%]
tests/integration/test_reranking.py::test_empty_candidates PASSED                                  [ 91%]
tests/integration/test_reranking.py::test_trace_records_before_after PASSED                        [ 94%]
tests/integration/test_reranking.py::test_patient_filter_survives PASSED                           [ 97%]
tests/integration/test_reranking.py::test_no_invented_cardiology_fact PASSED                       [100%]

========================================== 36 passed in 1.22s ==========================================
```