# Local LLM Integration with Ollama

## Purpose

The Ollama integration replaces the deterministic demonstration generator with
a real local language model while preserving the rest of the project:

```text
document ingestion
OCR
metadata extraction
chunking
embeddings
hybrid retrieval
reranking
context construction
citation validation
confidence gating
FastAPI
```

The project still uses its own `LLMClient` interface. Ollama is one replaceable
implementation rather than a dependency embedded throughout the codebase.

## Architecture

```text
RAGGenerator
    ↓ LLMRequest
OllamaLLMClient
    ↓ official Ollama Python client
http://127.0.0.1:11434/api/chat
    ↓
local model
    ↓ assistant content
LLMResponse
    ↓
citation and answer validation
```

## Files

```text
app/generation/ollama_llm_client.py
    Ollama adapter, configuration, readiness checks, and errors

app/generation/factory.py
    deterministic / ollama / auto backend selection

scripts/check_ollama.py
    server and installed-model check

scripts/run_ollama_rag.py
    complete local ingestion and RAG demonstration

tests/unit/test_ollama_llm_client.py
    request/response contract tests using a fake Ollama client

tests/integration/test_ollama_rag_generation.py
    end-to-end ingestion, retrieval, reranking, generation, and validation
```

## Ubuntu 24.04 installation

Install Ollama using its official Linux installer:

```bash
$ curl -fsSL https://ollama.com/install.sh | sh
```

Start the service:

```bash
$ sudo systemctl start ollama
$ sudo systemctl status ollama
```

A foreground development server can also be started with:

```bash
$ ollama serve
```

Install the Python integration:

```bash
$ source .venv/bin/activate
$ python -m pip install -r requirements-ollama.txt
```

## Pull a model

The default configuration uses:

```bash
$ ollama pull gemma3:4b
```

List installed models:

```bash
$ ollama list
```

Test the model directly:

```bash
$ ollama run gemma3:4b \
  "Reply with exactly: Ollama is ready."
```

## Hardware guidance

Model requirements vary by architecture, quantization, and context length.

A practical starting point for a small 4B-class quantized model is:

```text
CPU-only:
    16 GB system RAM recommended
    lower generation speed

GPU:
    enough VRAM to hold most or all model layers
    substantially faster generation

Disk:
    several GB per model
```

Larger models can require tens of GB of RAM or VRAM. Start with a small model,
measure quality and latency, and increase model size only when the evaluation
dataset shows a meaningful gain.

## Environment configuration

Copy the example:

```bash
$ cp .env.ollama.example .env
```

Load it:

```bash
$ set -a
$ source .env
$ set +a
```

Important variables:

```bash
$ export MDA_LLM_BACKEND=ollama
$ export MDA_OLLAMA_HOST=http://127.0.0.1:11434
$ export MDA_OLLAMA_MODEL=gemma3:4b
$ export MDA_OLLAMA_TIMEOUT_SECONDS=120
$ export MDA_OLLAMA_KEEP_ALIVE=5m
$ export MDA_OLLAMA_CONTEXT_LENGTH=8192
```

Retrieval can remain fully offline:

```bash
$ export MDA_EMBEDDING_BACKEND=hash
$ export MDA_STORAGE_BACKEND=local
$ export MDA_RERANKER_BACKEND=deterministic
```

Or use the model-backed retrieval stack:

```bash
$ export MDA_EMBEDDING_BACKEND=sentence-transformer
$ export MDA_STORAGE_BACKEND=chroma
$ export MDA_RERANKER_BACKEND=cross-encoder
```

## Readiness check

```bash
$ python scripts/check_ollama.py \
  --host http://127.0.0.1:11434 \
  --model gemma3:4b
```

Expected:

```text
Reachable:       True
Model available: True
```

When the model is missing, the script prints:

```bash
$ ollama pull gemma3:4b
```

## Run the complete Ollama workflow

```bash
$ python scripts/run_ollama_rag.py \
  data/samples \
  --host http://127.0.0.1:11434 \
  --model gemma3:4b \
  --embedding-backend hash \
  --storage-backend local \
  --reranker-backend deterministic \
  --reset
```

The command:

```text
checks Ollama readiness
ingests the native PDF and handwritten PNG
runs OCR where needed
indexes vector and BM25 chunks
retrieves and reranks evidence
generates through the local model
resolves citations
validates answer grounding
prints page-level sources
```

## Run FastAPI with Ollama

```bash
$ export MDA_LLM_BACKEND=ollama
$ export MDA_OLLAMA_HOST=http://127.0.0.1:11434
$ export MDA_OLLAMA_MODEL=gemma3:4b

$ uvicorn app.api.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Then use:

```text
http://127.0.0.1:8000/docs
```

The `/questions` endpoint returns HTTP 503 with a clear error when Ollama cannot
be reached, rather than returning an unhandled exception.

## Prompt and generation mapping

Project setting:

```python
GenerationConfig(
    temperature=0.0,
    max_tokens=600,
    stop_sequences=(),
)
```

Ollama options:

```python
{
    "temperature": 0.0,
    "num_predict": 600,
    "num_ctx": 8192,
}
```

The system and user prompts are sent as separate chat messages:

```python
[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
```

## Testing without a running model

The unit and integration tests inject a fake Ollama client.

This verifies:

```text
message roles
model name
temperature
token limit
context length
stop sequences
keep-alive
response metadata
citation resolution
full document pipeline
```

Run:

```bash
$ python -m pytest \
  tests/unit/test_ollama_llm_client.py \
  tests/integration/test_ollama_rag_generation.py \
  -v
```

=================== test session starts ===================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: langsmith-0.10.15, anyio-4.14.2
collected 6 items                                                                                                                                                         

tests/unit/test_ollama_llm_client.py::test_generate_maps_prompts_and_options PASSED                                                                                 [ 16%]
tests/unit/test_ollama_llm_client.py::test_status_detects_installed_model PASSED                                                                                    [ 33%]
tests/unit/test_ollama_llm_client.py::test_close_delegates_to_official_client PASSED                                                                                [ 50%]
tests/unit/test_ollama_llm_client.py::test_invalid_host_is_rejected PASSED                                                                                          [ 66%]
tests/unit/test_ollama_llm_client.py::test_connection_error_is_translated PASSED                                                                                    [ 83%]
tests/integration/test_ollama_rag_generation.py::test_ollama_adapter_generates_cited_answer_end_to_end PASSED                                                       [100%]

=================== 6 passed in 0.22s ===================

## Troubleshooting

### Connection refused

```bash
$ sudo systemctl status ollama
$ sudo systemctl start ollama
```

Or:

```bash
$ ollama serve
```

### Model not installed

```bash
$ ollama pull gemma3:4b
```

### Python package missing

```bash
$ python -m pip install -r requirements-ollama.txt
```

### Slow CPU generation

Use a smaller model or reduce:

```bash
$ export MDA_OLLAMA_CONTEXT_LENGTH=4096
```

### Out of memory

Stop other model workloads, choose a smaller quantized model, or reduce context
length. The retrieval pipeline should keep the final source count and context
budget small enough to avoid unnecessary prompt size.

## Security and privacy

Ollama keeps generation on the configured host, but local processing does not
automatically make a system compliant.

Continue to apply:

```text
authentication
authorization
disk encryption
least-privilege file access
log minimization
retention and deletion policies
malware scanning
patient isolation
formal privacy and security review
```