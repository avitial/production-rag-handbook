import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.ingestion.ocr import ocr_image

result = ocr_image(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200849_handwritten.png"
)

print("Extraction:", result.extraction_method)
print("OCR confidence:", result.confidence)
print("Patient:", result.metadata.get("patient_id"))
print("Document type:", result.metadata.get("document_type"))
print("Handwritten:", result.metadata.get("is_handwritten"))
print("Preprocessing:", result.metadata.get("preprocessing_operations"))
print(result.text)