import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook/app"))

from ingestion.chunking import (
    ChunkingConfig,
    ingest_and_chunk_local_path,
)

chunks = ingest_and_chunk_local_path(
    "/home/avitial/workspace/RAG/production-rag-handbook/data/development",
    config=ChunkingConfig(
        max_characters=300,
        overlap_characters=40,
    ),
)

chunk_id = 1
for chunk in chunks:
    print("=" * 70)
    print("chunk_id: ", chunk_id)
    print("File:", chunk.filename)
    print("Page:", chunk.page_number)
    print("Section:", chunk.section)
    print("Patient:", chunk.metadata.get("patient_id"))
    print("Format:", chunk.metadata.get("source_format"))
    print("Extraction:", chunk.metadata.get("extraction_method"))
    print(chunk.text)
    chunk_id+=1