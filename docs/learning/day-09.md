# Day 9 — Grounded Answer Generation

## Goal

Use reranked passages to generate concise answers with page-level citations and
explicit abstention when the documents do not support a claim.

The complete package still ingests local:

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
app/generation/llm_client.py
app/generation/local_llm_client.py
app/generation/context_builder.py
app/generation/prompts.py
app/generation/rag_generator.py
tests/unit/test_context_builder.py
tests/unit/test_prompt_builder.py
docs/learning/day-09.md
```

Supporting files:

```text
tests/integration/test_rag_generation.py
scripts/run_rag_generation.py
```

## Architecture

```text
Local documents
      ↓
Extraction and OCR
      ↓
Chunking and indexing
      ↓
Hybrid retrieval
      ↓
Cross-encoder reranking
      ↓
Context builder
      ↓
Grounded prompts
      ↓
LLM client
      ↓
Answer + resolved citations + diagnostics
```

## `llm_client.py`

Defines a backend-independent interface.

### Models

```text
GenerationConfig
LLMRequest
LLMResponse
LLMClient
```

The rest of the application does not depend on one provider's SDK.

A future hosted client can implement:

```python
class HostedClient(LLMClient):
    ...
```

without changing `RAGGenerator`.

## `local_llm_client.py`

Provides a deterministic offline generator.

It parses the structured source blocks, scores their relevance, extracts
answer-bearing lines, and emits numbered citations.

It is useful for:

- Tests
- Offline demonstrations
- Citation validation
- Abstention behavior
- Pipeline debugging

It is not a neural model and should not be used as a benchmark for production
answer quality.

## `context_builder.py`

Builds numbered source blocks:

```text
[SOURCE 1]
File: SYN-200849_handwritten.png
Page: 1
Section: Allergies
Patient ID: SYN-200849
Allergies
Latex
```

Responsibilities:

- Preserve reranked order
- Deduplicate chunk IDs
- Enforce source and character limits
- Avoid partial passages by default
- Build a citation map
- Report omitted and truncated sources

## `prompts.py`

The system prompt requires:

- Use only supplied evidence
- Do not mix patients
- Cite factual claims
- Abstain when evidence is missing
- Do not diagnose or recommend treatment
- Keep answers concise

The user prompt has explicit boundaries:

```text
QUESTION
INSTRUCTIONS
BEGIN CONTEXT
numbered sources
END CONTEXT
FINAL ANSWER
```

## `rag_generator.py`

Pseudo-code:

```text
build bounded context
build prompt bundle
call LLM client
parse [SOURCE N] markers
validate citation numbers
resolve citations to files and pages
detect abstention
return structured answer and diagnostics
```

The generator returns:

```text
answer text
resolved citations
built context
system and user prompts
raw LLM response
citation validation
abstention status
timing diagnostics
```

## Run fully offline

```bash
$ python scripts/run_rag_generation.py data/samples \
  --embedding-backend hash \
  --storage-backend local \
  --reranker-backend deterministic \
  --reset
```

This runs:

```text
native PDF extraction
handwritten PNG OCR
hybrid retrieval
deterministic reranking
deterministic grounded generation
```

## Questions

```text
What medication was prescribed at discharge?
What allergies are documented?
When is the follow-up appointment?
What was the primary diagnosis?
Which patient had a cardiology referral?
```

Expected behavior:

- Allergy and follow-up questions return cited evidence.
- Patient filters prevent cross-patient answers.
- Unsupported cardiology referral question abstains.
- The discharge wording may require abstention because medication presence does
  not prove it was prescribed at discharge.

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
  -v
```

## Test coverage

### Context builder

- Numbered sources
- File/page/patient provenance
- Deduplication
- Maximum sources
- Character budget
- Optional truncation
- Citation map

### Prompt builder

- Grounding instructions
- Citation requirement
- Patient separation
- Abstention wording
- Context boundaries
- Empty context handling

### RAG integration

- PDF and handwritten ingestion
- Allergy answer with citation
- Follow-up answer with citation
- Unsupported cardiology abstention
- Cross-patient prevention
- Context-budget diagnostics

## Limitations

The offline client is deterministic and rule-based.

A production implementation should add a concrete hosted or local neural-model
client, then evaluate:

- Faithfulness
- Citation correctness
- Answer relevance
- Abstention precision
- Latency
- Cost
- Prompt-injection resistance

## Acceptance criteria

- [ ] All required source formats still ingest.
- [ ] Reranked passages become numbered context.
- [ ] Context budgets are enforced.
- [ ] Prompts prohibit unsupported claims.
- [ ] Answers use numbered citations.
- [ ] Citation numbers resolve to files and pages.
- [ ] Patient filtering survives generation.
- [ ] Unsupported questions abstain.
- [ ] All synchronized tests pass.
- [ ] The example script runs end to end.

## Design questions

1. Why separate LLM clients from RAG orchestration?
2. Why number sources before generation?
3. Why validate citations after generation?
4. Why avoid partial passages by default?
5. Why preserve patient IDs in context headers?
6. Why is retrieval relevance not enough for answerability?
7. What should happen when the model cites an invalid source number?
8. How should context size be tuned?
9. Why use temperature zero for grounded QA?
10. What validation should Day 10 add?

