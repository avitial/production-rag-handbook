import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook/app"))

from ingestion.pdf_extractor import extract_pdf


result = extract_pdf(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200989.pdf"
)

print("Document ID:", result.document_id)
print("SHA-256:", result.source_hash)
print("Pages:", result.total_pages)

for page, diagnostics in zip(
    result.pages,
    result.diagnostics,
):
    print("=" * 70)
    print("Page:", page.page_number)
    print("Patient:", page.metadata.patient_id)
    print("Method:", page.extraction_method)
    print("Characters:", diagnostics.character_count)
    print("Meaningful:", diagnostics.meaningful_character_count)
    print("Needs OCR:", diagnostics.needs_ocr)
    print(page.text)