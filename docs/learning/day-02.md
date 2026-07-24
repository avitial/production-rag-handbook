# Day 2 — Local Document Ingestion and Chunking

## Goal

Load local PDFs and images, including handwritten-note images, convert them
into page models, and create section-aware chunks with page-level provenance.

## Supported sources

```text
PDF
JPEG
JPG
PNG
Handwritten-note images
```

The `SYN-200989.pdf` file is a synthetic, one-page medical record and is
used as the first local PDF example.

## Dependencies

```bash
pip install pydantic pytest pymupdf pillow pytesseract
```

Tesseract is an external application.

Windows:

```powershell
winget install UB-Mannheim.TesseractOCR
```

Ubuntu:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

macOS:

```bash
brew install tesseract
```

## Deliverable 1: `app/domain/models.py`

Includes:

- `SourceFormat`
- `ExtractionMethod`
- `DocumentType`
- `DocumentMetadata`
- `LocalSource`
- `DocumentPage`
- `DocumentChunk`

The models preserve:

```text
source path
filename
file format
document ID
source hash
page number
extraction method
patient ID
handwritten status
section
character offsets
```

## Deliverable 2: `app/ingestion/chunking.py`

Includes both minimal Day 2 local ingestion and chunking.

### Local discovery

```python
from app.ingestion.chunking import discover_local_sources

sources = discover_local_sources("data/development")
```

The function accepts a single file or directory.

### Ingest one local source

```python
from app.ingestion.chunking import ingest_local_source

pages = ingest_local_source(sources[0])
```

### Ingest a local path

```python
from app.ingestion.chunking import ingest_local_path

pages = ingest_local_path(
    "data/development",
    recursive=True,
)
```

### Ingest and chunk directly

```python
from app.ingestion.chunking import (
    ChunkingConfig,
    ingest_and_chunk_local_path,
)

chunks = ingest_and_chunk_local_path(
    "/data/development/SYN-200989.pdf",
    config=ChunkingConfig(
        max_characters=300,
        overlap_characters=40,
    ),
)
```

## PDF extraction behavior

For each PDF page:

1. Try PyMuPDF native extraction.
2. Check whether enough meaningful text was extracted.
3. If not, render the page as an image.
4. OCR the rendered page with Tesseract.
5. Preserve the PDF page number.

The supplied example should use `NATIVE_PDF` because it contains embedded text.

## Image behavior

For JPEG, JPG, and PNG:

1. Correct EXIF orientation.
2. Convert to grayscale.
3. Apply automatic contrast.
4. OCR with Tesseract.
5. Store the image as page 1.

## Handwritten-note behavior

An image is marked as handwritten when:

- Its filename contains `handwritten` or `handwriting`, or
- The caller constructs `LocalSource(..., is_handwritten=True)`.

Handwritten notes receive:

```text
source_format = handwritten_note
extraction_method = ocr_handwritten
metadata.is_handwritten = true
```

The OCR path is functional, but Tesseract may not read difficult handwriting
well. Test with clear block handwriting first.

## Patient ID extraction

The baseline recognizes text such as:

```text
Patient ID: SYN-200989
```

and stores it in `DocumentMetadata.patient_id`.

Later, metadata extraction should move into a dedicated module and cover
document type and date.

## Section-aware chunking

The supplied PDF contains headings such as:

```text
Patient Demographics
Clinical Notes (SOAP)
Medications
Allergies
Problems
Vital Signs
Procedures & Results
```

The section detector keeps each heading with its content.

## Run against the example PDF

From the project root:

```python
from app.ingestion.chunking import (
    ChunkingConfig,
    ingest_and_chunk_local_path,
)

chunks = ingest_and_chunk_local_path(
    "/mnt/data/SYN-200989.pdf",
    config=ChunkingConfig(
        max_characters=300,
        overlap_characters=40,
    ),
)

for chunk in chunks:
    print("=" * 70)
    print("File:", chunk.filename)
    print("Page:", chunk.page_number)
    print("Section:", chunk.section)
    print("Patient:", chunk.metadata.get("patient_id"))
    print("Method:", chunk.metadata.get("extraction_method"))
    print(chunk.text)
```

You should see content containing:

```text
SYN-200989
Metformin
Shellfish
Hypertension
HbA1c: 6.8%
Knee X-ray
```

## Run against a local directory

Example directory:

```text
data/sample_documents/
├── SYN-200989.pdf
├── scanned_lab_report.pdf
├── medication_list.jpg
├── discharge_form.jpeg
├── intake_form.png
└── handwritten_note_01.png
```

Run:

```python
chunks = ingest_and_chunk_local_path(
    "data/development",
    recursive=True,
)
```

## Unit tests

Run:

```bash
pytest tests/unit/test_chunking.py -v
```

The test suite includes:

- Missing-file validation
- PDF/JPEG/JPG/PNG discovery
- Handwritten filename recognition
- OCR-image ingestion using a deterministic fake OCR function
- Patient ID extraction
- Section detection
- Chunk overlap
- Exact offset reconstruction
- Validation of offset spans
- Ingestion and chunking of `SYN-200989.pdf`

## Hands-on exercises

### Exercise 1

Copy `SYN-200989.pdf` into a local sample directory and ingest the directory.

### Exercise 2

Convert one synthetic note to PNG and compare OCR output with PDF extraction.

### Exercise 3

Create a clear handwritten note containing:

```text
Patient ID: SYN-HAND-001
PLAN
Follow up in two weeks.
```

Save it as:

```text
handwritten_note_01.png
```

Inspect the extracted text and handwritten metadata.

### Exercise 4

Try chunk configurations:

```text
300/40
500/75
800/150
```

Record chunk count and readability.

### Exercise 5

Lower `native_text_min_characters` and observe when a sparse PDF page uses
native extraction versus OCR.

## Acceptance criteria

- [ ] The supplied PDF ingests successfully.
- [ ] Its patient ID is extracted as `SYN-200989`.
- [ ] Native PDF extraction is recorded.
- [ ] JPEG, JPG, and PNG sources are discovered.
- [ ] Image OCR can be invoked.
- [ ] Handwritten images are marked separately.
- [ ] Chunks preserve page number and source path.
- [ ] Chunks preserve section and patient metadata.
- [ ] Chunk offsets reconstruct their exact source text.
- [ ] Unit tests pass.
- [ ] You can explain why handwriting OCR is less reliable.
