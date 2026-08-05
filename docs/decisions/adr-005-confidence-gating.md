# ADR-005: Gate Answers Using Explicit Confidence Features


## Context

The RAG pipeline now retrieves, reranks, generates, cites, validates, and
serializes answers. It still needs a final policy that determines whether an
answer should be returned.

A single vector similarity is not answer confidence. High similarity can still
occur when:

- The passage is related but does not answer the question.
- The wrong patient is retrieved.
- The generator cites an invalid source.
- The answer is unsupported.
- The output JSON is malformed.

## Decision

Confidence gating will use an explicit feature object and a deterministic
policy.

Outcomes:

```text
ACCEPT
ABSTAIN
REJECT
```

### ACCEPT

All hard validation requirements pass and configured evidence thresholds are
met.

### ABSTAIN

The pipeline is functioning, but evidence is insufficient or a configured
ranking threshold is not met.

### REJECT

A correctness or integrity check failed, such as invalid citations, ungrounded
answer, validation issues, or invalid JSON.

## Features

The baseline records:

```text
retrieval candidate count
top retrieval similarity
retrieval similarity margin
reranked candidate count
top reranker score
reranker score margin
context source count
citation count
citation validity
grounding validity
JSON validity
abstention state
patient-filter state
invalid-citation count
validation-issue count
```

## Rule ordering

Hard integrity failures are evaluated first:

```text
invalid JSON
invalid citations
ungrounded answer
validation issues
```

Then evidence sufficiency:

```text
generator abstention
too little context
too few citations
retrieval threshold
reranker threshold
```

## Score thresholds

Thresholds are optional by default because score ranges differ across:

- Embedding models
- Vector-store distance functions
- Reranker models
- Offline deterministic backends
- Datasets

Thresholds must be calibrated on an evaluation set.

## Logging and metrics

Every decision should record:

- Question identifier or trace ID
- Decision
- Reason codes
- Confidence features
- Latencies
- Patient-filter status

Logs should avoid full sensitive documents.

Metrics include:

```text
accepted answers
abstentions
rejections
validation failures
average retrieval latency
average reranking latency
average generation latency
```

## Alternatives considered

### Use top vector similarity only

Rejected because it does not measure answerability or generation integrity.

### Use the reranker score only

Rejected because a reranker ranks candidates but does not validate citations or
generated claims.

### Ask the LLM for a confidence percentage

Rejected because self-reported confidence is not calibrated evidence.

### Always return the generated answer

Rejected because safe abstention is a core requirement.

## Consequences

### Positive

- Decisions are explainable.
- Integrity failures are separated from low evidence.
- Thresholds are configurable.
- Logs and metrics support calibration.
- Tests remain deterministic.

### Limitations

- Thresholds require dataset-specific tuning.
- Feature rules are not a calibrated probability.
- Lexical grounding validation remains approximate.
- Some correct answers may be conservatively abstained.

## Validation

Tests verify:

- Valid answers are accepted.
- Generator abstentions remain abstentions.
- Invalid citations are rejected.
- Ungrounded answers are rejected.
- Missing context causes abstention.
- Retrieval and reranker thresholds can gate answers.
- Invalid JSON is rejected.
- End-to-end decisions are logged and measured.

## Revisit criteria

Revisit when:

- Ragas or another evaluator supplies calibrated metrics.
- More evaluation examples are available.
- Model or embedding backends change.
- Confidence calibration is required.
- A production telemetry backend is introduced.
