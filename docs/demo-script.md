# Day 14 Demo Script

## 1. Start offline mode

```bash
source .venv/bin/activate

export MDA_EMBEDDING_BACKEND=hash
export MDA_STORAGE_BACKEND=local
export MDA_RERANKER_BACKEND=deterministic

uvicorn app.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## 2. Check health

```bash
curl -s http://127.0.0.1:8000/health \
  | python -m json.tool
```

## 3. Ingest the sample PDF and handwritten image

```bash
curl -s -X POST \
  http://127.0.0.1:8000/documents/ingest-local \
  -H 'Content-Type: application/json' \
  -d '{"source_path":"data/samples"}' \
  | python -m json.tool
```

Explain native PDF extraction, OCR, section chunking, embeddings, metadata, and
the BM25 refresh.

## 4. Demonstrate hybrid search

```bash
curl -s -X POST \
  http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"Latex allergy",
    "filters":{"patient_id":"SYN-200849"},
    "top_k":5
  }' | python -m json.tool
```

Show patient isolation, scores, filename, page, and section.

## 5. Ask an answerable question

```bash
curl -s -X POST \
  http://127.0.0.1:8000/questions \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"What allergies are documented?",
    "filters":{"patient_id":"SYN-200849"},
    "candidate_k":10,
    "final_k":5
  }' | python -m json.tool
```

Expected:

```text
decision: accept
answer: Latex with [SOURCE N]
citation: handwritten PNG, page 1
validation: passed
```

## 6. Demonstrate abstention

```bash
curl -s -X POST \
  http://127.0.0.1:8000/questions \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"Which patient had a cardiology referral?",
    "candidate_k":10,
    "final_k":5
  }' | python -m json.tool
```

Expected:

```text
decision: abstain
reason: generator_abstained
```

## 7. Demonstrate upload

```bash
curl -s -X POST \
  http://127.0.0.1:8000/documents/upload \
  -F 'file=@data/samples/handwritten/SYN-200849_handwritten.png' \
  | python -m json.tool
```

## Interview close

> I built an end-to-end retrieval pipeline for native PDFs, scans, images, and
> handwritten notes. It uses section-aware chunks, vector and BM25 retrieval,
> reranking, grounded generation, page citations, validation, confidence
> gating, evaluation, tuning, and a documented FastAPI interface.
