import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.ingestion.pdf_extractor import extract_pdf_pages
from app.ingestion.chunking import (
    ChunkingConfig,
    chunk_pages,
)

pages = extract_pdf_pages(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development/SYN-200989.pdf"
)

chunks = chunk_pages(
    pages,
    config=ChunkingConfig(
        max_characters=300,
        overlap_characters=40,
    ),
)

for chunk in chunks:
    print("=" * 60)
    print("Page:", chunk.page_number)
    print("Section:", chunk.section)
    print(chunk.text)
