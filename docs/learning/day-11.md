# Day 11 — Confidence Gating, Logging, and Metrics

## Goal

Decide whether a validated RAG answer should be accepted, safely abstained, or
rejected.

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
app/confidence/features.py
app/confidence/policy.py
app/observability/logger.py
app/observability/metrics.py
tests/unit/test_confidence_policy.py
docs/decisions/adr-005-confidence-gating.md
docs/learning/day-11.md
```

Supporting files:

```text
tests/integration/test_confidence_gating.py
scripts/run_confidence_gated_rag.py
```

## Architecture

```text
Ingestion and OCR
      ↓
Hybrid retrieval
      ↓
Reranking
      ↓
Grounded generation
      ↓
Citation, answer, and JSON validation
      ↓
Confidence feature extraction
      ↓
Confidence policy
      ├── ACCEPT
      ├── ABSTAIN
      └── REJECT
      ↓
Structured logs and metrics
```

## `features.py`

Creates one immutable feature object from:

```text
retrieval results
reranker results
generated answer
answer validation
JSON validation
```

It calculates rank-score margins safely and records validation state.

## `policy.py`

Hard failures produce `REJECT`.

Insufficient evidence produces `ABSTAIN`.

Only complete and validated answers produce `ACCEPT`.

Pseudo-code:

```text
if JSON/citations/grounding failed:
    REJECT
elif generator abstained:
    ABSTAIN
elif context or citations are insufficient:
    ABSTAIN
elif optional score thresholds fail:
    ABSTAIN
else:
    ACCEPT
```

## `logger.py`

Writes JSONL events such as:

```json
{
  "event": "confidence_decision",
  "question": "What allergies are documented?",
  "decision": "accept",
  "reason_codes": ["all_requirements_met"]
}
```

Full source documents are not logged by default.

## `metrics.py`

Tracks:

```text
request count
accepted answers
abstentions
rejections
retrieval latency
reranking latency
generation latency
feature gauges
```

The collector is dependency-free and can later be replaced by Prometheus or
OpenTelemetry exporters.

## Run end to end

```bash
$ python scripts/run_confidence_gated_rag.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reranker-backend deterministic \
  --reset
```

```text
Ingestion
  processed: 2
  skipped:   0
  failed:    0
  indexed:   18

Q: What medication was prescribed at discharge?
Decision: ABSTAIN
Answer: I could not find enough explicit evidence in the provided sources to answer this question.
Reasons: generator_abstained

Q: What allergies are documented?
Decision: ACCEPT
Answer: Section: Allergies Allergies Latex [SOURCE 1]
Reasons: all_requirements_met

Q: When is the follow-up appointment?
Decision: ACCEPT
Answer: S: Patient reports routine follow-up with mild intermittent symptoms. P: Continue current therapy, encourage healthy lifestyle, follow-up in 6 months. [SOURCE 1]
Reasons: all_requirements_met

Q: What was the primary diagnosis?
Decision: ACCEPT
Answer: Section: Problems Problems Active Diagnosis: Hypertension [SOURCE 1]
Reasons: all_requirements_met

Q: Which patient had a cardiology referral?
Decision: ABSTAIN
Answer: I could not find enough explicit evidence in the provided sources to answer this question.
Reasons: generator_abstained

Metrics
{
  "counters": {
    "requests_total": 5,
    "answers_abstain_total": 2,
    "answers_accept_total": 3
  },
  "gauges": {
    "last_context_source_count": 5.0
  },
  "latencies": {
    "retrieval_ms": {
      "count": 5,
      "minimum_ms": 0.6272678729146719,
      "maximum_ms": 1.1312898714095354,
      "average_ms": 0.8390335366129875
    },
    "reranking_ms": {
      "count": 5,
      "minimum_ms": 0.2024141140282154,
      "maximum_ms": 0.2632210962474346,
      "average_ms": 0.23222644813358784
    },
    "generation_ms": {
      "count": 5,
      "minimum_ms": 0.20407000556588173,
      "maximum_ms": 0.37419400177896023,
      "average_ms": 0.2818802837282419
    }
  }
}
```

Expected behavior:

```text
Discharge medication → ABSTAIN
Allergy question      → ACCEPT
Follow-up question    → ACCEPT
Primary diagnosis     → ACCEPT
Cardiology referral   → ABSTAIN
```

## Run all tests

```bash
$ python -m pytest \
  tests/integration/test_chroma_ingestion.py \
  tests/integration/test_vector_retrieval.py \
  tests/unit/test_tokenizer.py \
  tests/unit/test_fusion.py \
  tests/integration/test_hybrid_retrieval.py \
  tests/integration/test_reranking.py \
  tests/unit/test_context_builder.py \
  tests/unit/test_prompt_builder.py \
  tests/integration/test_rag_generation.py \
  tests/unit/test_citation_validator.py \
  tests/unit/test_answer_validator.py \
  tests/unit/test_json_validator.py \
  tests/integration/test_validated_rag.py \
  tests/unit/test_confidence_policy.py \
  tests/integration/test_confidence_gating.py \
  -v
```

```text
================================================= test session starts =================================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: langsmith-0.10.15, anyio-4.14.2
collected 74 items                                                                                                                                                        

tests/integration/test_chroma_ingestion.py::test_end_to_end_ingestion PASSED                                                                                        [  1%]
tests/integration/test_chroma_ingestion.py::test_metadata_and_documents_are_persisted PASSED                                                                        [  2%]
tests/integration/test_chroma_ingestion.py::test_second_run_is_skipped PASSED                                                                                       [  4%]
tests/integration/test_chroma_ingestion.py::test_changed_chunking_configuration_rebuilds PASSED                                                                     [  5%]
tests/integration/test_chroma_ingestion.py::test_all_sample_questions_return_ranked_results PASSED                                                                  [  6%]
tests/integration/test_chroma_ingestion.py::test_persistence_survives_new_store_instance PASSED                                                                     [  8%]
tests/integration/test_vector_retrieval.py::test_filter_builder_no_filters PASSED                                                                                   [  9%]
tests/integration/test_vector_retrieval.py::test_filter_builder_combines_patient_and_page PASSED                                                                    [ 10%]
tests/integration/test_vector_retrieval.py::test_filter_builder_date_range PASSED                                                                                   [ 12%]
tests/integration/test_vector_retrieval.py::test_unfiltered_retrieval_returns_ranked_passages PASSED                                                                [ 13%]
tests/integration/test_vector_retrieval.py::test_patient_filter_excludes_other_patient PASSED                                                                       [ 14%]
tests/integration/test_vector_retrieval.py::test_nonexistent_patient_returns_empty PASSED                                                                           [ 16%]
tests/integration/test_vector_retrieval.py::test_filename_filter PASSED                                                                                             [ 17%]
tests/integration/test_vector_retrieval.py::test_citation_label PASSED                                                                                              [ 18%]
tests/integration/test_vector_retrieval.py::test_similarity_conversion PASSED                                                                                       [ 20%]
tests/unit/test_tokenizer.py::test_preserves_medical_identifiers_and_values PASSED                                                                                  [ 21%]
tests/unit/test_tokenizer.py::test_removes_common_stop_words_by_default PASSED                                                                                      [ 22%]
tests/unit/test_tokenizer.py::test_stop_word_removal_can_be_disabled PASSED                                                                                         [ 24%]
tests/unit/test_tokenizer.py::test_unicode_is_normalized PASSED                                                                                                     [ 25%]
tests/unit/test_tokenizer.py::test_tokenization_is_deterministic PASSED                                                                                             [ 27%]
tests/unit/test_fusion.py::test_duplicate_chunk_is_fused_once PASSED                                                                                                [ 28%]
tests/unit/test_fusion.py::test_top_n_limits_results PASSED                                                                                                         [ 29%]
tests/unit/test_fusion.py::test_ranked_conversion_marks_hybrid_metadata PASSED                                                                                      [ 31%]
tests/unit/test_fusion.py::test_invalid_fusion_constant_is_rejected PASSED                                                                                          [ 32%]
tests/integration/test_hybrid_retrieval.py::test_bm25_rebuilds_from_ingested_chunks PASSED                                                                          [ 33%]
tests/integration/test_hybrid_retrieval.py::test_bm25_finds_exact_patient_identifier PASSED                                                                         [ 35%]
tests/integration/test_hybrid_retrieval.py::test_hybrid_retrieval_returns_unique_ranked_chunks PASSED                                                               [ 36%]
tests/integration/test_hybrid_retrieval.py::test_hybrid_patient_filter_prevents_cross_patient_results PASSED                                                        [ 37%]
tests/integration/test_hybrid_retrieval.py::test_keyword_and_hybrid_find_exact_allergy PASSED                                                                       [ 39%]
tests/integration/test_hybrid_retrieval.py::test_unanswerable_cardiology_query_does_not_create_false_metadata PASSED                                                [ 40%]
tests/integration/test_reranking.py::test_reranker_promotes_exact_allergy_passage PASSED                                                                            [ 41%]
tests/integration/test_reranking.py::test_provenance_survives PASSED                                                                                                [ 43%]
tests/integration/test_reranking.py::test_empty_candidates PASSED                                                                                                   [ 44%]
tests/integration/test_reranking.py::test_trace_records_before_after PASSED                                                                                         [ 45%]
tests/integration/test_reranking.py::test_patient_filter_survives PASSED                                                                                            [ 47%]
tests/integration/test_reranking.py::test_no_invented_cardiology_fact PASSED                                                                                        [ 48%]
tests/unit/test_context_builder.py::test_builds_numbered_sources_with_provenance PASSED                                                                             [ 50%]
tests/unit/test_context_builder.py::test_duplicate_chunks_are_included_once PASSED                                                                                  [ 51%]
tests/unit/test_context_builder.py::test_source_limit_is_enforced PASSED                                                                                            [ 52%]
tests/unit/test_context_builder.py::test_character_budget_stops_before_partial_source PASSED                                                                        [ 54%]
tests/unit/test_context_builder.py::test_optional_truncation_marks_context PASSED                                                                                   [ 55%]
tests/unit/test_prompt_builder.py::test_system_prompt_requires_grounding_and_citations PASSED                                                                       [ 56%]
tests/unit/test_prompt_builder.py::test_user_prompt_contains_question_and_context_boundaries PASSED                                                                 [ 58%]
tests/unit/test_prompt_builder.py::test_empty_context_is_explicit PASSED                                                                                            [ 59%]
tests/unit/test_prompt_builder.py::test_prompt_bundle_preserves_custom_system_prompt PASSED                                                                         [ 60%]
tests/integration/test_rag_generation.py::test_generates_allergy_answer_with_valid_citation PASSED                                                                  [ 62%]
tests/integration/test_rag_generation.py::test_generates_follow_up_answer_from_handwritten_source PASSED                                                            [ 63%]
tests/integration/test_rag_generation.py::test_unsupported_cardiology_question_abstains PASSED                                                                      [ 64%]
tests/integration/test_rag_generation.py::test_patient_filter_prevents_cross_patient_answer PASSED                                                                  [ 66%]
tests/integration/test_rag_generation.py::test_context_budget_and_source_limit_are_reported PASSED                                                                  [ 67%]
tests/integration/test_rag_generation.py::test_discharge_qualified_medication_question_abstains PASSED                                                              [ 68%]
tests/unit/test_citation_validator.py::test_parses_unique_citations_in_order PASSED                                                                                 [ 70%]
tests/unit/test_citation_validator.py::test_valid_citation_passes PASSED                                                                                            [ 71%]
tests/unit/test_citation_validator.py::test_invalid_source_number_is_rejected PASSED                                                                                [ 72%]
tests/unit/test_citation_validator.py::test_non_abstaining_answer_requires_citation PASSED                                                                          [ 74%]
tests/unit/test_citation_validator.py::test_abstention_does_not_require_citation PASSED                                                                             [ 75%]
tests/unit/test_citation_validator.py::test_unresolved_valid_source_is_reported PASSED                                                                              [ 77%]
tests/unit/test_answer_validator.py::test_grounded_answer_passes PASSED                                                                                             [ 78%]
tests/unit/test_answer_validator.py::test_missing_citation_fails PASSED                                                                                             [ 79%]
tests/unit/test_answer_validator.py::test_cross_patient_citation_fails PASSED                                                                                       [ 81%]
tests/unit/test_answer_validator.py::test_unsupported_answer_without_overlap_fails PASSED                                                                           [ 82%]
tests/unit/test_answer_validator.py::test_abstention_passes_without_citation PASSED                                                                                 [ 83%]
tests/unit/test_json_validator.py::test_valid_schema_round_trips PASSED                                                                                             [ 85%]
tests/unit/test_json_validator.py::test_missing_required_fields_fail PASSED                                                                                         [ 86%]
tests/unit/test_json_validator.py::test_non_serializable_value_fails PASSED                                                                                         [ 87%]
tests/integration/test_validated_rag.py::test_validated_allergy_response_round_trips PASSED                                                                         [ 89%]
tests/integration/test_confidence_gating.py::test_accepts_when_all_requirements_pass PASSED                                                                         [ 90%]
tests/integration/test_confidence_gating.py::test_generator_abstention_stays_abstained PASSED                                                                       [ 91%]
tests/integration/test_confidence_gating.py::test_invalid_citations_are_rejected PASSED                                                                             [ 93%]
tests/integration/test_confidence_gating.py::test_ungrounded_answer_is_rejected PASSED                                                                              [ 94%]
tests/integration/test_confidence_gating.py::test_missing_context_causes_abstention PASSED                                                                          [ 95%]
tests/integration/test_confidence_gating.py::test_retrieval_threshold_causes_abstention PASSED                                                                      [ 97%]
tests/integration/test_confidence_gating.py::test_reranker_threshold_causes_abstention PASSED                                                                       [ 98%]
tests/integration/test_confidence_gating.py::test_invalid_json_is_rejected PASSED                                                                                   [100%]

================================================= 74 passed in 1.73s =================================================
```

## Acceptance criteria

- [ ] Required local formats still ingest.
- [ ] Confidence features are extracted.
- [ ] Valid grounded answers are accepted.
- [ ] Evidence-insufficient answers abstain.
- [ ] Integrity failures are rejected.
- [ ] Decision reasons are machine-readable.
- [ ] Structured logs are written.
- [ ] Metrics summarize decisions and latency.
- [ ] All synchronized tests pass.
- [ ] The example script runs end to end.

## Design questions

1. Why is vector similarity not answer confidence?
2. What is the difference between abstain and reject?
3. Why evaluate integrity failures before score thresholds?
4. How should thresholds be calibrated?
5. Why record score margins?
6. Why preserve reason codes?
7. Which metrics indicate over-abstention?
8. Which metrics indicate unsafe acceptance?
9. Why avoid logging full documents?
10. How will Day 12 evaluate the policy?
