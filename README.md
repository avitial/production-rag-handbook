# production-rag-handbook
### End-to-End Retrieval-Augmented Generation (RAG) Pipeline for Unstructured Medical Documents

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](#)

Production Rag Handbook (Medical Document Assistant) is an end-to-end Retrieval-Augmented Generation (RAG) application that processes unstructured medical documents from local sources. The system ingests PDFs, scanned documents, images, and handwritten notes, extracts structured information using OCR, indexes documents into a vector database, performs hybrid retrieval, reranks relevant passages, and generates citation-grounded answers through a REST API.

This project was developed as a learning platform for modern AI document processing systems while following production-inspired software architecture and engineering practices.

---

# Key Features

- 📄 Ingest native PDF medical documents
- 🖼️ Process scanned PDFs and image-based documents
- ✍️ Perform OCR on handwritten medical notes
- 🏥 Extract document metadata (Patient ID, document type, page numbers, sections)
- ✂️ Section-aware document chunking with configurable overlap
- 🧠 Generate embeddings using Sentence Transformers
- 🗄️ Store embeddings in ChromaDB
- 🔍 Hybrid retrieval using Vector Search + BM25
- 📊 Reciprocal Rank Fusion (RRF) for result fusion
- 🎯 Cross-Encoder reranking support
- 🤖 Retrieval-Augmented Generation (RAG)
- 📚 Citation-grounded answers with document/page references
- ✅ Citation, JSON, and answer validation
- 📈 Confidence scoring with Accept / Abstain / Reject decisions
- 📋 Structured logging and retrieval metrics
- 🧪 Retrieval evaluation and tuning framework
- 🚀 FastAPI REST API with automatic Swagger documentation
- 🔌 Modular architecture designed for production-ready AI pipelines

---

# Tech Stack

## Programming Language

- Python 3.12+

## AI / Machine Learning

- Sentence Transformers
- ChromaDB
- BM25 Retrieval
- CrossEncoder
- Retrieval-Augmented Generation (RAG)

## OCR

- Tesseract OCR
- pytesseract

## API Framework

- FastAPI
- Uvicorn
- Pydantic

## Document Processing

- PyPDF
- Pillow

## Storage

- ChromaDB
- SQLite (Document Registry)

## Testing

- pytest

## Evaluation

- Ragas (optional)
- Custom retrieval metrics
- MRR
- Hit Rate
- Precision
- Recall
- nDCG

---

# Getting Started

## Prerequisites

Before installing the project, ensure the following software is available.

### Operating System

Recommended

- Ubuntu 24.04 LTS

Compatible

- Windows 11
- macOS
- Other Linux distributions

### Software

Install the following:

- Python 3.12+
- Git
- pip
- Tesseract OCR

Ubuntu installation:

```bash
$ sudo apt update

$ sudo apt install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    tesseract-ocr \
    tesseract-ocr-eng
```

Verify installation:

```bash
python3.12 --version

tesseract --version
```

---

## Installation

### Clone the repository

```bash
$ git clone https://github.com/avitial/production-rag-handbook.git

cd production-rag-handbook
```

### Create a virtual environment

```bash
$ python3.12 -m venv .venv
```

Activate it

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

### Upgrade pip

```bash
$ python -m pip install --upgrade pip setuptools wheel
```

### Install project dependencies

```bash
$ python -m pip install -r requirements.txt
```

---

## Configuration

The project can run completely offline or use production components.

### Offline Mode (Recommended)

```bash
export MDA_EMBEDDING_BACKEND=hash
export MDA_STORAGE_BACKEND=local
export MDA_RERANKER_BACKEND=deterministic
```

### Production Mode

```bash
export MDA_EMBEDDING_BACKEND=sentence-transformer
export MDA_STORAGE_BACKEND=chroma
export MDA_RERANKER_BACKEND=cross-encoder
```

### Optional Runtime Configuration

```bash
export MDA_CHROMA_DIR=runtime/chroma
export MDA_REGISTRY_PATH=runtime/document-registry.sqlite3
export MDA_UPLOAD_DIR=runtime/uploads
export MDA_LOG_PATH=runtime/logs/events.jsonl
export MDA_COLLECTION_NAME=medical_documents
```

---

# Usage

## Start the API

```bash
$ uvicorn app.api.main:app \
    --host 127.0.0.1 \
    --port 8000
```

Interactive API documentation

```
http://127.0.0.1:8000/docs
```

Alternative API documentation

```
http://127.0.0.1:8000/redoc
```

---

## Ingest Local Documents

```bash
curl -X POST \
http://127.0.0.1:8000/documents/ingest-local \
-H "Content-Type: application/json" \
-d '{
  "source_path":"data/samples"
}'
```

---

## Upload a Document

```bash
curl -X POST \
http://127.0.0.1:8000/documents/upload \
-F "file=@data/samples/handwritten/SYN-200849_handwritten.png"
```

---

## Search Documents

```bash
curl -X POST \
http://127.0.0.1:8000/search \
-H "Content-Type: application/json" \
-d '{
  "query":"Latex allergy",
  "filters":{
      "patient_id":"SYN-200849"
  },
  "top_k":5
}'
```

---

## Ask Questions

```bash
curl -X POST \
http://127.0.0.1:8000/questions \
-H "Content-Type: application/json" \
-d '{
    "question":"What allergies are documented?",
    "filters":{
        "patient_id":"SYN-200849"
    },
    "candidate_k":10,
    "final_k":5
}'
```

Example response:

```json
{
  "decision": "accept",
  "answer": "The documented allergy is Latex.",
  "citations": [
    {
      "filename": "SYN-200849_handwritten.png",
      "page_number": 1
    }
  ]
}
```
## Local LLM Generation with Ollama

The project can use a real local language model through Ollama while keeping
documents, prompts, and generated answers on the configured machine.

### Install Ollama on Ubuntu 24.04

```bash
curl -fsSL https://ollama.com/install.sh | sh

$ sudo systemctl start ollama
$ sudo systemctl status ollama
```

Install the project's optional integration dependencies:

```bash
$ source .venv/bin/activate
$ python -m pip install -r requirements-ollama.txt
```

Pull the default model:

```bash
$ ollama pull gemma3:4b
```

Check readiness:

```bash
$ python scripts/check_ollama.py \
  --model gemma3:4b
```

Configure the application:

```bash
$ export MDA_LLM_BACKEND=ollama
$ export MDA_OLLAMA_HOST=http://127.0.0.1:11434
$ export MDA_OLLAMA_MODEL=gemma3:4b
$ export MDA_OLLAMA_TIMEOUT_SECONDS=120
$ export MDA_OLLAMA_KEEP_ALIVE=5m
$ export MDA_OLLAMA_CONTEXT_LENGTH=8192
```

Run the direct end-to-end demonstration:

```bash
$ python scripts/run_ollama_rag.py \
  data/samples \
  --model gemma3:4b \
  --reset
```

Or start FastAPI with Ollama generation:

```bash
$ uvicorn app.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

See [`docs/ollama-integration.md`](docs/ollama-integration.md) for architecture,
hardware guidance, testing, and troubleshooting.

---

# Project Workflow

```text
Local Documents

├── Native PDFs

├── Scanned PDFs

├── JPEG / JPG

└── PNG / Handwritten Notes

        │

        ▼

OCR / Native Text Extraction

        ▼

Metadata Extraction

        ▼

Section-Based Chunking

        ▼

Embedding Generation

        ▼

Vector Storage

        ▼

Hybrid Retrieval

(Vector Search + BM25)

        ▼

Reciprocal Rank Fusion

        ▼

CrossEncoder Reranking

        ▼

Context Builder

        ▼

Citation-Grounded Generation

        ▼

Validation

        ▼

Confidence Policy

        ▼

REST API Response
```

---

# Roadmap

Future improvements planned for the project include:

- Docker support
- Docker Compose deployment
- Kubernetes deployment
- Local LLM integration (Ollama)
- Multi-document conversational memory
- Incremental indexing
- Streaming API responses
- User authentication and authorization
- Role-Based Access Control (RBAC)
- Multi-language OCR
- Clinical entity extraction
- FHIR interoperability
- DICOM image support
- Cloud storage integrations (AWS S3, Azure Blob, Google Cloud Storage)
- CI/CD pipeline with GitHub Actions
- Expanded evaluation datasets and benchmarking

---

# Contributing

Contributions are welcome!

If you would like to improve the project:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/my-feature
```

3. Commit your changes.

```bash
git commit -m "Add my new feature"
```

4. Push the branch.

```bash
git push origin feature/my-feature
```

5. Open a Pull Request describing your changes.

Please use GitHub Issues to report bugs, request features, or discuss improvements before implementing major changes.

---

# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for complete details.

---

# Acknowledgements

Production Rag Handbook (Medical Document Assistant) was built as an educational and portfolio project to demonstrate modern Retrieval-Augmented Generation (RAG) concepts and production-inspired AI software engineering practices.

The project integrates OCR, document processing, semantic search, vector databases, hybrid retrieval, reranking, grounded generation, confidence scoring, evaluation, and REST APIs into a modular architecture suitable for experimentation, learning, and future extension.
