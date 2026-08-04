# Day 10 — Grounding Validation, Citation Checks, and Structured JSON

## Goal

Validate generated RAG answers before they are returned to downstream systems.

The complete package continues to support local ingestion of:

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
app/api/schemas.py
app/validation/citation_validator.py
app/validation/answer_validator.py
app/validation/json_validator.py
tests/unit/test_citation_validator.py
tests/unit/test_answer_validator.py
docs/decisions/adr-004-grounding-and-citations.md
docs/learning/day-10.md
```

Supporting files:

```text
tests/unit/test_json_validator.py
tests/integration/test_validated_rag.py
scripts/run_validated_rag.py
```

## Architecture

```text
Local documents
      ↓
Ingestion and OCR
      ↓
Hybrid retrieval
      ↓
Reranking
      ↓
Grounded generation
      ↓
Citation validation
      ↓
Answer grounding validation
      ↓
API schema construction
      ↓
JSON validation and serialization
```

## `app/api/schemas.py`

Defines transport models for:

```text
ingestion requests
ingestion responses
query filters
RAG query requests
citations
validation issues
validation summaries
RAG answer responses
error responses
```

The schemas prevent internal implementation details from leaking into the API
contract.

## `citation_validator.py`

Checks:

```text
[SOURCE N] syntax
source number exists
source number resolves
non-abstaining answers have citations
```

Pseudo-code:

```text
parse unique source numbers
compare with context citation map
compare with resolved citations
add missing or invalid citation issues
return typed result
```

## `answer_validator.py`

Checks:

```text
empty answer
citation validity
patient isolation
context existence
minimum evidence overlap
abstention behavior
```

Patient isolation is validated after generation, even when retrieval already
used a patient filter.

## `json_validator.py`

Performs:

```text
dataclass-to-dictionary conversion
recursive JSON-safe conversion
required-field validation
field-type validation
citation-shape validation
JSON serialization
JSON deserialization
round-trip comparison
```

## Run the offline validated workflow

```bash
$ python scripts/run_validated_rag.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reranker-backend deterministic \
  --reset
```

The script prints:

```text
answer
citations
citation validity
answer grounding status
JSON validity
final structured JSON
```

## Expected question behavior

### Medication prescribed at discharge

The documents list medication but do not explicitly say it was prescribed at
discharge.

Expected:

```text
abstain
citation validation passes
answer validation passes
JSON validation passes
```

### Allergies for SYN-200849

Expected:

```text
Latex
citation to handwritten PNG page 1
no Shellfish leakage
all validations pass
```

### Follow-up appointment

Expected:

```text
6 months
citation to handwritten source
all validations pass
```

### Primary diagnosis for SYN-200989

Expected:

```text
Hypertension
citation to PDF page 1
all validations pass
```

### Cardiology referral

No explicit referral exists.

Expected:

```text
abstain
all validations pass
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
  -v
```

## Acceptance criteria

- [ ] Required local formats still ingest.
- [ ] Valid citations pass.
- [ ] Invalid source numbers fail.
- [ ] Missing citations fail for factual answers.
- [ ] Abstentions can omit citations.
- [ ] Cross-patient citations fail.
- [ ] Unsupported claims are flagged.
- [ ] API response schemas are stable.
- [ ] JSON round-trip validation passes.
- [ ] All synchronized tests pass.
- [ ] The example script runs end to end.

## Design questions

1. Why are prompts not enough to guarantee grounding?
2. Why validate citations after generation?
3. Why validate patient identity again after retrieval?
4. Why can a citation be syntactically valid but semantically weak?
5. Why is lexical overlap only a baseline?
6. What should an API do with validation failures?
7. Why use typed response schemas?
8. Why test JSON round trips?
9. Should abstentions require citations?
10. What claim-level validation should be added next?