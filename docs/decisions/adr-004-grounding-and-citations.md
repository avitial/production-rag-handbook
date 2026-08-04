## Context

Day 9 generates answers from reranked passages and resolves numbered citations.
Generation alone is not sufficient for a production-oriented RAG pipeline.

A model can still:

- Cite a source number not present in context
- Return no citations
- Reference a different patient's document
- Make a claim with little evidence overlap
- Produce malformed structured output
- Return a valid-looking answer that should have abstained

The application needs deterministic checks after generation and before sending
the response downstream.

## Decision

The pipeline will validate three layers:

```text
1. Citation validity
2. Answer grounding behavior
3. JSON response structure
```

The API boundary will use typed dataclasses in `app/api/schemas.py`.

## Citation validation

A citation is valid when:

- It uses `[SOURCE N]`.
- Source `N` exists in the context citation map.
- Source `N` resolves into the response citation list.
- A non-abstaining factual answer includes at least one valid citation.

Abstaining answers are not required to cite a source.

## Answer grounding validation

The deterministic baseline checks:

- Empty answer
- Missing citations
- Cross-patient citations
- Answer without context
- Minimum lexical evidence overlap
- Correct abstention handling

Lexical overlap is only a baseline signal. It does not prove semantic
faithfulness or clinical correctness.

## JSON validation

The API response must serialize and deserialize without loss.

Required top-level fields:

```text
question
answer
citations
abstained
validation
diagnostics
```

Citation objects must include at least:

```text
source_number
filename
page_number
```

## Failure behavior

Validation issues are returned as machine-readable codes.

Examples:

```text
invalid_source_number
unresolved_source_number
missing_citation
cross_patient_citation
insufficient_evidence_overlap
not_json_serializable
missing_field
invalid_type
```

The pipeline should not silently discard validation failures.

A production API may:

- Return HTTP 422 for malformed client input.
- Return a validated abstention when evidence is insufficient.
- Return HTTP 500 or a controlled error when internal serialization fails.
- Log the trace ID for debugging.

## Patient isolation

When a query is filtered to one patient, every resolved citation must belong to
that patient.

This check is performed after generation because the answer layer should not
assume retrieval filtering was implemented correctly.

## Alternatives considered

### Trust prompt instructions only

Rejected because prompts are not enforcement mechanisms.

### Validate citations inside the LLM prompt only

Rejected because generated markers must be checked against actual context.

### Use only model-based faithfulness evaluation

Deferred because it adds cost, nondeterminism, and another model dependency.

### Require citations for abstentions

Rejected because an abstention may reflect the absence of explicit evidence.

### Return untyped dictionaries

Rejected because field drift and serialization errors become harder to detect.

## Consequences

### Positive

- Invalid source markers are detected.
- Cross-patient leakage is surfaced.
- Downstream JSON has a stable shape.
- Validation issues are machine-readable.
- Tests remain deterministic and offline.
- API and pipeline responsibilities are separated.

### Limitations

- Lexical overlap can miss valid paraphrases.
- Lexical overlap can pass copied but misleading text.
- Validation does not replace medical review.
- Citation presence does not prove claim-level support.
- More advanced faithfulness evaluation is still needed.

## Validation

Tests verify:

- Valid citations pass.
- Invalid source numbers fail.
- Missing citations fail.
- Abstentions can omit citations.
- Cross-patient citations fail.
- Unsupported claims with no overlap fail.
- Valid API responses round-trip through JSON.
- Missing fields and nonserializable values fail.
- End-to-end answers are validated and serialized.

## Revisit criteria

Revisit this decision when:

- Claim-level citation validation is required.
- A model-based faithfulness evaluator is introduced.
- FastAPI or another web framework becomes the API layer.
- Pydantic schemas replace dataclass transport models.
- Patient-isolation rules expand beyond patient ID.
- Streaming responses are added.
