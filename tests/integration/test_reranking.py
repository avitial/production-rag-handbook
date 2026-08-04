from pathlib import Path
import shutil
from PIL import Image
from app.embeddings.sentence_transformer import DeterministicHashEmbeddingProvider
from app.ingestion.pipeline import IngestionConfig,LocalIngestionPipeline
from app.observability.retrieval_trace import RetrievalTraceWriter,create_retrieval_trace
from app.reranking.base import RerankingRequest
from app.reranking.cross_encoder import DeterministicReranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import SearchFilters,VectorSearchRequest
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry
HAND=Path('/home/avitial/workspace/RAG/production-rag-handbook/data/samples/handwritten/SYN-200849_handwritten.png'); PDF=Path('/home/avitial/workspace/RAG/production-rag-handbook/data/samples/native-pdf/SYN-200989.pdf')

def fake_ocr(_image:Image.Image,*,language:str)->str:
    return '*** SYNTHETIC MEDICAL RECORD ***\nPatient Demographics\nPatient ID: SYN-200849\nClinical Notes (SOAP)\nA: Routine exam\nP: Follow-up appointment in 6 months.\nMedications\nCurrent: Metformin\nAllergies\nLatex\nProblems\nActive Diagnosis: Routine exam\n'

def samples(tmp):
    d=tmp/'samples'; d.mkdir(); shutil.copy2(HAND,d/'SYN-200849_handwritten.png') if HAND.exists() else Image.new('RGB',(200,200),'white').save(d/'SYN-200849_handwritten.png')
    if PDF.exists(): shutil.copy2(PDF,d/'SYN-200989.pdf')
    return d

def system(tmp):
    provider=DeterministicHashEmbeddingProvider(dimensions=128)
    store=ChromaStore(embedding_provider=provider,collection_name='day8',persistence_directory=tmp/'vectors',backend='local')
    ingestion=LocalIngestionPipeline(chroma_store=store,registry=DocumentRegistry(tmp/'registry.sqlite3'),config=IngestionConfig(max_characters=300,overlap_characters=40),ocr_function=fake_ocr)
    vector=VectorRetriever(store); bm25=BM25Retriever(store); hybrid=HybridRetriever(vector_retriever=vector,bm25_retriever=bm25)
    return ingestion,hybrid,DeterministicReranker()

def test_reranker_promotes_exact_allergy_passage(tmp_path):
    ingestion,hybrid,reranker=system(tmp_path); assert ingestion.ingest(samples(tmp_path)).failed_files==0; hybrid.rebuild_keyword_index()
    retrieved=hybrid.search(VectorSearchRequest('What allergies are documented?',10,SearchFilters(patient_id='SYN-200849')))
    reranked=reranker.rerank(RerankingRequest('What allergies are documented?',retrieved.results,3))
    assert 'Latex' in reranked.results[0].passage.text and reranked.results[0].passage.patient_id=='SYN-200849'

def test_provenance_survives(tmp_path):
    ingestion,hybrid,reranker=system(tmp_path); ingestion.ingest(samples(tmp_path)); hybrid.rebuild_keyword_index()
    retrieved=hybrid.search(VectorSearchRequest('What was the primary diagnosis?',8,SearchFilters(patient_id='SYN-200989')))
    reranked=reranker.rerank(RerankingRequest('What was the primary diagnosis?',retrieved.results,3))
    first=reranked.results[0]; assert first.passage.filename=='SYN-200989.pdf' and first.passage.page_number==1 and 'page 1' in first.citation_label()

def test_empty_candidates():
    r=DeterministicReranker().rerank(RerankingRequest('question',(),3)); assert r.results==() and r.diagnostics.candidate_count==0

def test_trace_records_before_after(tmp_path):
    ingestion,hybrid,reranker=system(tmp_path); ingestion.ingest(samples(tmp_path)); hybrid.rebuild_keyword_index()
    retrieved=hybrid.search(VectorSearchRequest('What allergies are documented?',6)); reranked=reranker.rerank(RerankingRequest('What allergies are documented?',retrieved.results,3))
    writer=RetrievalTraceWriter(tmp_path/'traces.jsonl'); writer.write(create_retrieval_trace(retrieval_response=retrieved,reranking_response=reranked,metadata={'case':'allergy'}))
    rec=writer.read_all()[0]; assert rec['final_result_count']==3 and any(c['rerank_rank'] is not None for c in rec['candidates'])

def test_patient_filter_survives(tmp_path):
    ingestion,hybrid,reranker=system(tmp_path); ingestion.ingest(samples(tmp_path)); hybrid.rebuild_keyword_index()
    retrieved=hybrid.search(VectorSearchRequest('What medication is documented?',10,SearchFilters(patient_id='SYN-200849')))
    reranked=reranker.rerank(RerankingRequest('What medication is documented?',retrieved.results,5))
    assert reranked.results and all(x.passage.patient_id=='SYN-200849' for x in reranked.results)

def test_no_invented_cardiology_fact(tmp_path):
    ingestion,hybrid,reranker=system(tmp_path); ingestion.ingest(samples(tmp_path)); hybrid.rebuild_keyword_index(); q='Which patient had a cardiology referral?'
    reranked=reranker.rerank(RerankingRequest(q,hybrid.search(VectorSearchRequest(q,10)).results,5))
    assert all('cardiology referral' not in x.passage.text.lower() for x in reranked.results)
