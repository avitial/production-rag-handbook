"""Ingest local samples and run Day 6 vector retrieval end to end."""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from app.embeddings.sentence_transformer import create_embedding_provider
from app.ingestion.pipeline import IngestionConfig,LocalIngestionPipeline
from app.retrieval.models import SearchFilters,VectorSearchRequest
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry
QUESTIONS=[
 'What medication was prescribed at discharge?',
 'What allergies are documented?',
 'When is the follow-up appointment?',
 'What was the primary diagnosis?',
 'Which patient had a cardiology referral?',
]
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('source',nargs='?',default='data/samples')
 p.add_argument('--embedding-backend',choices=['auto','sentence-transformer','hash'],default='auto')
 p.add_argument('--storage-backend',choices=['auto','chroma','local'],default='auto')
 p.add_argument('--patient-id'); p.add_argument('--top-k',type=int,default=3); p.add_argument('--reset',action='store_true')
 p.add_argument('--chroma-dir',default='./chroma_data'); p.add_argument('--registry',default='./data/document_registry.sqlite3')
 return p.parse_args()
def main()->int:
 a=parse_args()
 if a.reset: shutil.rmtree(a.chroma_dir,ignore_errors=True); Path(a.registry).unlink(missing_ok=True)
 provider=create_embedding_provider(a.embedding_backend)
 store=ChromaStore(embedding_provider=provider,persistence_directory=a.chroma_dir,backend=a.storage_backend)
 registry=DocumentRegistry(a.registry)
 ingestion=LocalIngestionPipeline(chroma_store=store,registry=registry,config=IngestionConfig())
 report=ingestion.ingest(a.source)
 print('Ingestion'); print(f'  processed: {report.processed_files}'); print(f'  skipped:   {report.skipped_files}'); print(f'  failed:    {report.failed_files}'); print(f'  indexed:   {report.indexed_chunk_count}')
 for warning in report.warnings: print(f'  warning: {warning}')
 if report.failed_files: return 2
 retriever=VectorRetriever(store); filters=SearchFilters(patient_id=a.patient_id)
 print('\nVector retrieval')
 for q in QUESTIONS:
  response=retriever.search(VectorSearchRequest(query=q,top_k=a.top_k,filters=filters))
  print(f'\nQ: {q}'); print('  filter:',response.diagnostics.where_filter)
  if not response.results: print('  No matching passages.'); continue
  for item in response.results:
   preview=' '.join(item.text.split())[:130]
   print(f'  {item.rank}. similarity={item.similarity:.4f} patient={item.patient_id or "unknown"} source={item.citation_label()} | {preview}')
 return 0
if __name__=='__main__': raise SystemExit(main())
