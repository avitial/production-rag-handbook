import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.ingestion.ocr import ocr_pdf

results = ocr_pdf(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200849_scanned.pdf"
)

for result in results:
    print("=" * 70)
    print("Page:", result.page_number)
    print("Method:", result.extraction_method)
    print("OCR fallback:", result.metadata.get("ocr_fallback_used"))
    print("Patient:", result.metadata.get("patient_id"))
    print(result.text)