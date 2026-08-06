# Privacy and Security

This is a synthetic-data educational project, not a certified medical system.

## Included upload controls

- `.pdf`, `.jpeg`, `.jpg`, and `.png` allowlist
- configurable maximum upload size
- removal of client path components
- sanitized basename
- randomized server-side filename
- deletion of uploads when ingestion fails

These controls are a baseline. Production also requires file-signature checks,
malware scanning, and archive-bomb protections.

## Required before real medical data

```text
TLS
authentication
role-based authorization
patient and tenant access rules
encrypted storage and backups
secret management
audit logging
retention and deletion policy
rate limiting
security tests
incident response
legal and HIPAA review
```

The local-path ingestion endpoint should be restricted to trusted
administrators or removed.

## Data minimization

- Use synthetic or properly de-identified records.
- Send only required excerpts to models.
- Do not log complete records.
- Keep logs access-controlled and short-lived.
- Do not commit `.env`, keys, or real records.

## Patient isolation

Metadata filters are applied during retrieval, and resolved citations are
checked again after generation. This is defense in depth, not authorization.

## Prompt injection

Retrieved documents are untrusted. Context is delimited, document instructions
are treated as data, and outputs are validated. A production system should add
adversarial testing and prevent retrieved text from triggering tools.

## Medical safety

The assistant must not diagnose or recommend treatment. Answers should quote or
summarize documented facts, cite pages, and abstain when evidence is absent.
