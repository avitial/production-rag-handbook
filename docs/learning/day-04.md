# Day 4 — OCR, Image Preprocessing, and Metadata Extraction

## Goal

Add OCR fallback for scanned PDFs and direct OCR for JPEG, JPG, PNG, andhandwritten-note images.

The supplied sample:
- SYN-200849_handwritten.png

It contains synthetic patient ID SYN-200849, Vietnamese preferred language,Metformin, Latex allergy, vital signs, LDL result, and lumbar MRI information.

## Architecture

Local source
    ↓
PDF or image dispatch
    ↓
Native PDF text check
    ├─ usable → preserve native text
    └─ sparse → render page
                    ↓
              image preprocessing
                    ↓
                 Tesseract
                    ↓
          OCR text + confidence
                    ↓
          metadata extraction

## Dependencies
$ pip install -r requirements.txt

Python packages:
pydantic>=2.0
pymupdf
pillow
pytesseract
pytest

Tesseract is also an external executable.

Ubuntu:
$ sudo apt-get install tesseract-ocr

macOS:
$ brew install tesseract

Windows:
$ winget install UB-Mannheim.TesseractOCR

## image_preprocessor.py; Main methods:

### load_image()

Pseudo-code:
- resolve path
- validate extension
- open image
- load pixels
- return independent image copy

### preprocess_image()

Pseudo-code:
- copy input
- correct EXIF orientation
- convert to grayscale
- upscale small image
- apply autocontrast
- sharpen edges
- optionally threshold
- return processed image and operation list

### preprocess_image_file()
Combines file loading and preprocessing.

## ocr.py

### ocr_image()
Used for JPEG, JPG, PNG, and handwritten-note images.

from app.ingestion.ocr import ocr_image

result = ocr_image(
    "data/samples/handwritten/SYN-200849_handwritten.png"
)

print(result.text)
print(result.confidence)
print(result.metadata)

### ocr_pdf()
For every PDF page:

extract native text
if sufficient:
    return native page
else:
    render page
    preprocess
    OCR

### ingest_ocr_source()
Dispatches based on extension:

results = ingest_ocr_source("local-file.pdf")
results = ingest_ocr_source("local-image.png")

metadata_extractor.py

Extracts:
- Patient ID
- Document date
- Provider
- Preferred language
- Broad document type
- Handwritten status

Each rule-based match stores evidence:
- field name
- captured value
- matched source text
- start offset
- end offset

Example:

from app.ingestion.metadata_extractor import extract_metadata

result = extract_metadata(
    "Patient ID: SYN-200849\n"
    "Preferred Language: Vietnamese\n"
    "Clinical Notes (SOAP)"
)

print(result.metadata.patient_id)
print(result.metadata.preferred_language)
print(result.metadata.document_type)

### OCR confidence

Tesseract returns word-level confidence values.

Rows containing -1 are ignored, then valid values are averaged.

This score estimates OCR recognition quality. It is not:

- Retrieval confidence
- Answer confidence
- Medical correctness
- A calibrated probability

### Run the handwritten sample

from app.ingestion.ocr import ocr_image

result = ocr_image(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200849_handwritten.png"
)

print("Method:", result.extraction_method)
print("OCR confidence:", result.confidence)
print("Patient:", result.metadata.get("patient_id"))
print("Handwritten:", result.metadata.get("is_handwritten"))
print(result.text)

Expected robust keywords include:

- synthetic medical record
- SYN-200849
- Metformin
- Latex
- 129/91
- LDL
- MRI Lumbar

OCR output may differ slightly by Tesseract version and platform.

### Scanned PDF sample
The package creates:

/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200849_scanned.pdf

This PDF contains the sample image as a page without a native text layer.

Run:

from app.ingestion.ocr import ocr_pdf

results = ocr_pdf(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200849_scanned.pdf"
)

for page in results:
    print(page.page_number)
    print(page.extraction_method)
    print(page.text)

## Tests

$ pytest tests/unit/test_metadata_extraction.py -v
================================== test session starts ====================================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python3
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: anyio-4.14.2
collected 13 items                                                                                                                                

../tests/unit/test_metadata_extraction.py::test_extracts_patient_id_language_date_and_type PASSED                    [  7%]
../tests/unit/test_metadata_extraction.py::test_evidence_offsets_match_original_normalized_text PASSED               [ 15%]
../tests/unit/test_metadata_extraction.py::test_handwritten_flag_forces_handwritten_document_type PASSED             [ 23%]
../tests/unit/test_metadata_extraction.py::test_missing_patient_and_date_add_warnings PASSED                         [ 30%]
../tests/unit/test_metadata_extraction.py::test_document_type_inference[DISCHARGE SUMMARY-discharge_summary] PASSED  [ 38%]
../tests/unit/test_metadata_extraction.py::test_document_type_inference[Laboratory Results-lab_report] PASSED        [ 46%]
../tests/unit/test_metadata_extraction.py::test_document_type_inference[Imaging: MRI lumbar-imaging_report] PASSED   [ 53%]
../tests/unit/test_metadata_extraction.py::test_document_type_inference[Clinical Notes (SOAP)-clinical_note] PASSED  [ 61%]
../tests/unit/test_metadata_extraction.py::test_document_type_inference[Prescription-prescription] PASSED            [ 69%]
../tests/unit/test_metadata_extraction.py::test_document_type_inference[Referral-referral]            [ 76%]
../tests/unit/test_metadata_extraction.py::test_document_type_inference[Unknown form-other]           [ 84%]
../tests/unit/test_metadata_extraction.py::test_invalid_date_returns_none PASSED                      [ 92%]
../tests/unit/test_metadata_extraction.py::test_normalize_text_collapses_repeated_blank_lines PASSED  [100%]
=========================================== 13 passed in 0.10s ==========================================

$ pytest tests/integration/test_ocr.py -v
========================================== test session starts ==========================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/avitial/workspace/venv/rag/bin/python3
cachedir: .pytest_cache
rootdir: /home/avitial/workspace/RAG/production-rag-handbook
plugins: anyio-4.14.2
collected 7 items                                                                                                                                

../tests/integration/test_ocr.py::test_preprocessor_upscales_and_grayscales PASSED               [ 14%]
../tests/integration/test_ocr.py::test_average_confidence_ignores_negative_rows PASSED           [ 28%]
../tests/integration/test_ocr.py::test_handwritten_image_ocr_with_fake_engine PASSED             [ 42%]
../tests/integration/test_ocr.py::test_dispatches_png_to_image_ocr PASSED                        [ 57%]
../tests/integration/test_ocr.py::test_native_pdf_skips_ocr PASSED                               [ 71%]
../tests/integration/test_ocr.py::test_scanned_pdf_uses_ocr_fallback PASSED                      [ 85%]
../tests/integration/test_ocr.py::test_real_tesseract_extracts_key_sample_content PASSED         [100%]

=========================================== 7 passed in 1.88s =============================================

Tests include:
- Patient ID extraction
- Preferred-language extraction
- Date parsing
- Document-type inference
- Evidence offsets
- Missing-field warnings
- Image preprocessing
- Upscaling
- Confidence averaging
- Handwritten OCR classification
- PDF native-text bypass
- Scanned-PDF fallback
- Real Tesseract sample extraction

## Exercises
- Compare OCR with and without upscaling.
- Try thresholds 140, 180, and 210.
- Compare Tesseract page segmentation modes 3, 6, and 11.
- Create a low-contrast copy and compare results.
- Rotate the sample and verify EXIF/orientation handling.
- Add a date field and confirm metadata extraction.
- Test a clear handwritten block-letter note.
- Record every OCR error and classify it as character, spacing, layout, or field error.

## Acceptance criteria
- PNG sample preprocesses successfully.
- JPEG and JPG are supported.
- Handwritten status is preserved.
- Scanned PDF uses OCR fallback.
- Native PDF bypasses OCR.
- Patient ID SYN-200849 is extracted.
- Preferred language is extracted.
- OCR confidence is recorded.
- Metadata evidence includes offsets.
- Unit and integration tests pass.
- You can explain why OCR confidence is not answer confidence.