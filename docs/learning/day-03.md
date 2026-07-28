# Day 3 — Local File Validation and Native PDF Extraction

## Goal

Build a reliable extraction layer that validates local sources, creates stabledocument identities, extracts native PDF text page by page, and flags sparsepages for OCR on Day 4.

## Supported local formats

The validator recognizes:
- .pdf
- .jpeg
- .jpg
- .png
- handwritten-note images


On Day 3 we extract native PDF text. Image, scanned-PDF, and handwriting OCR are validated here but implemented on Day 4.

## Deliverables

### app/ingestion/pdf_extractor.py

Includes:
- PDFExtractionConfig
- PDFPageDiagnostics
- PDFExtractionResult
- normalize_pdf_text()
- count_meaningful_characters()
- page_needs_ocr()
- extract_patient_id()
- extract_pdf()
- extract_pdf_pages()

### app/ingestion/file_validator.py

Includes:
- Supported-extension mapping
- File existence checks
- Regular-file checks
- Empty-file rejection
- Configurable size limit
- Basic PDF/PNG/JPEG signature checks
- Handwritten-note filename recognition
- Single-file validation
- Recursive directory discovery

### app/utils/hashing.py

Includes:
- Streaming SHA-256
- Local file hashing
- Short hash prefixes
- Deterministic document IDs

### tests/integration/test_pdf_extraction.py

Tests:
- Deterministic hashing
- Stable document IDs
- PDF validation
- Signature mismatch rejection
- PDF/PNG/JPG/JPEG discovery
- Native extraction of SYN-200989.pdf
- Patient ID extraction
- Expected medical text
- OCR-candidate detection
- Text normalization
- Non-PDF rejection

### data/samples/native-pdf/

Contains the synthetic SYN-200989.pdf example and a README warning againstreal patient data.

### Architecture

Local path
   ↓
File validation
   ↓
SHA-256 identity
   ↓
PyMuPDF page extraction
   ↓
Text normalization
   ↓
Page diagnostics
   ↓
DocumentPage objects
   ↓
Day 2 chunking


## Install

$ pip install -r requirements-day-03.txt

Run extraction:

from app.ingestion.pdf_extractor import extract_pdf

result = extract_pdf(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200989.pdf"
)

print(result.document_id)
print(result.source_hash)
print(result.total_pages)

for page, diagnostics in zip(result.pages, result.diagnostics):
    print("Page:", page.page_number)
    print("Patient:", page.metadata.patient_id)
    print("Characters:", diagnostics.character_count)
    print("Needs OCR:", diagnostics.needs_ocr)
    print(page.text)

Expected content includes:

- SYN-200989
- Quinn Kim
- Metformin
- Shellfish
- Hypertension
- 109/95 mmHg
- HbA1c: 6.8%
- Knee X-ray

### Why native extraction first?

Native text is generally:
- Faster than OCR
- More accurate when embedded text exists
- Easier to reproduce
- Less likely to introduce spelling errors

Sparse pages are flagged for Day 4 OCR rather than automatically trusted.

### OCR-candidate pseudo-code:

meaningful = count letters and numbers
if meaningful < threshold:
    needs_ocr = true
else:
    needs_ocr = false

This is only a heuristic. A title page may be sparse without being scanned.

### Run tests
$ pytest tests/integration/test_pdf_extraction.py -v

============================================= test session starts ===================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python3
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: anyio-4.14.2
collected 10 items                                                                                                                                                        

tests/integration/test_pdf_extraction.py::test_hash_is_deterministic PASSED      [ 10%]
tests/integration/test_pdf_extraction.py::test_validator_accepts_pdf PASSED      [ 20%]
tests/integration/test_pdf_extraction.py::test_validator_rejects_fake_pdf PASSED [ 30%]
tests/integration/test_pdf_extraction.py::test_discovery_supports_pdf_png_jpg_jpeg PASSED  [ 40%]
tests/integration/test_pdf_extraction.py::test_extract_uploaded_pdf PASSED       [ 50%]
tests/integration/test_pdf_extraction.py::test_pages_wrapper PASSED              [ 60%]
tests/integration/test_pdf_extraction.py::test_sparse_text_detection PASSED      [ 70%]
tests/integration/test_pdf_extraction.py::test_meaningful_count PASSED           [ 80%]
tests/integration/test_pdf_extraction.py::test_normalization PASSED              [ 90%]
tests/integration/test_pdf_extraction.py::test_non_pdf_rejected PASSED           [100%]


### Exercises
- Rename the PDF and verify its hash and document ID remain unchanged.
- Modify a copy and verify its hash changes.
- Compare sort_text=True and sort_text=False.
- Raise the OCR threshold to 1,000 and observe the diagnostic.
- Create a fake .pdf text file and confirm validation rejects it.
- Print and inspect result.pdf_metadata.

### Acceptance criteria
- The sample PDF validates.
- Hashing is deterministic.
- A stable document ID is produced.
- One native page is extracted.
- Patient ID SYN-200989 is found.
- Medical fields appear in the extracted text.
- Page diagnostics are recorded.
- The page is not marked for OCR at the default threshold.
- PDF/PNG/JPG/JPEG discovery works.
- Integration tests pass.