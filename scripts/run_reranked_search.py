from __future__ import annotations
import argparse,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from app.embeddings.sentence_transformer import create_embedding_provider
from app.ingestion.pipeline import IngestionConfig,LocalIngestionPipeline
from app.observability.retrieval_trace import RetrievalTraceWriter,create_retrieval_trace
from app.reranking.base import RerankingRequest
from app.reranking.cross_encoder import create_reranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import SearchFilters,VectorSearchRequest
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry
Q=['What medication was prescribed at discharge?','What allergies are documented?','When is the follow-up appointment?','What was the primary diagnosis?','Which patient had a cardiology referral?']
def args():
 p=argparse.ArgumentParser(); p.add_argument('source',nargs='?',default='data/samples'); p.add_argument('--embedding-backend',choices=['auto','sentence-transformer','hash'],default='auto'); p.add_argument('--storage-backend',choices=['auto','chroma','local'],default='auto'); p.add_argument('--reranker-backend',choices=['auto','cross-encoder','deterministic'],default='auto'); p.add_argument('--patient-id'); p.add_argument('--candidate-k',type=int,default=10); p.add_argument('--final-k',type=int,default=3); p.add_argument('--reset',action='store_true'); p.add_argument('--chroma-dir',default='./chroma_data'); p.add_argument('--registry',default='./data/document_registry.sqlite3'); p.add_argument('--trace-file',default='./logs/retrieval-traces.jsonl'); return p.parse_args()
def main():
 a=args();
 if a.reset: shutil.rmtree(a.chroma_dir,ignore_errors=True); Path(a.registry).unlink(missing_ok=True); Path(a.trace_file).unlink(missing_ok=True)
 provider=create_embedding_provider(a.embedding_backend); store=ChromaStore(embedding_provider=provider,persistence_directory=a.chroma_dir,backend=a.storage_backend); registry=DocumentRegistry(a.registry)
 report=LocalIngestionPipeline(chroma_store=store,registry=registry,config=IngestionConfig()).ingest(a.source)
 print('Ingestion'); print('  processed:',report.processed_files); print('  skipped:  ',report.skipped_files); print('  failed:   ',report.failed_files); print('  indexed:  ',report.indexed_chunk_count)
 if report.failed_files:
  [print('  warning:',x) for x in report.warnings]; return 2
 vector=VectorRetriever(store); bm25=BM25Retriever(store); bm25.rebuild(); hybrid=HybridRetriever(vector_retriever=vector,bm25_retriever=bm25); reranker=create_reranker(a.reranker_backend); writer=RetrievalTraceWriter(a.trace_file); filters=SearchFilters(patient_id=a.patient_id)
 print('Reranker:',reranker.model_name)
 for q in Q:
  t=time.perf_counter(); retrieved=hybrid.search(VectorSearchRequest(q,a.candidate_k,filters)); r_ms=(time.perf_counter()-t)*1000; reranked=reranker.rerank(RerankingRequest(q,retrieved.results,a.final_k)); writer.write(create_retrieval_trace(retrieval_response=retrieved,reranking_response=reranked,retrieval_duration_ms=r_ms))
  print('\nQ:',q)
  if not reranked.results: print('  No passages.'); continue
  for x in reranked.results:
   p=x.passage; print(f'  {x.rerank_rank}. rerank={x.rerank_score:.4f} original_rank={x.original_rank} patient={p.patient_id or "unknown"} {p.citation_label()} | {" ".join(p.text.split())[:130]}')
 print('\nTrace file:',Path(a.trace_file).resolve()); return 0
if __name__=='__main__': raise SystemExit(main())
