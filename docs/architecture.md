# Medical Document Assistant Architecture

## System context

```text
Client
  ↓ HTTP
FastAPI
  ├── /documents
  ├── /search
  ├── /questions
  └── /health
  ↓
RuntimeServices
  ├── extraction and OCR
  ├── chunking and metadata
  ├── embeddings and vector store
  ├── BM25 and hybrid retrieval
  ├── reranking
  ├── grounded generation
  ├── validation and confidence policy
  └── logging and metrics
```

## Ingestion flow

```text
PDF / JPEG / JPG / PNG
        ↓
format validation
        ↓
native PDF extraction or OCR
        ↓
patient and source metadata
        ↓
page-bounded section chunks with overlap
        ↓
embeddings and persistent vectors
        ↓
BM25 index rebuild
```

## Search flow

```text
query + patient/document filters
        ↓
vector retrieval + BM25
        ↓
Reciprocal Rank Fusion
        ↓
ranked passages with page provenance
```

## Question flow

```text
question
  ↓
hybrid candidates
  ↓
reranker
  ↓
bounded numbered context
  ↓
grounded generator
  ↓
citation, answer, and JSON validation
  ↓
confidence policy
  ↓
ACCEPT / ABSTAIN / REJECT
```

## API modules

```text
app/api/main.py
    app factory and router registration

app/api/dependencies.py
    settings and shared service graph

app/api/routes/documents.py
    local and multipart ingestion

app/api/routes/search.py
    hybrid search

app/api/routes/questions.py
    confidence-gated RAG

app/api/routes/health.py
    health and readiness
```

## Runtime modes

Offline:

```text
hash embeddings
local vector storage
deterministic reranker
deterministic grounded generator
```

Model-backed:

```text
Sentence Transformer embeddings
ChromaDB
CrossEncoder reranking
replaceable LLMClient
```

## Default persistence

```text
runtime/chroma/
runtime/document-registry.sqlite3
runtime/uploads/
runtime/logs/api-events.jsonl
```
