from __future__ import annotations
from datetime import date
from pathlib import Path
import shutil
from PIL import Image
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.embeddings.sentence_transformer import DeterministicHashEmbeddingProvider
from app.ingestion.pipeline import IngestionConfig,LocalIngestionPipeline
from app.retrieval.filter_builder import build_chroma_where
from app.retrieval.models import SearchFilters,VectorSearchRequest
from app.retrieval.vector_retriever import VectorRetriever,cosine_distance_to_similarity
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry
HAND=Path('/home/avitial/workspace/RAG/production-rag-handbook/data/samples/handwritten/SYN-200849_handwritten.png'); PDF=Path('/home/avitial/workspace/RAG/production-rag-handbook/data/samples/native-pdf/SYN-200989.pdf')
def fake_ocr(_image:Image.Image,*,language:str)->str:
 return 'Patient Demographics\nPatient ID: SYN-200849\nClinical Notes (SOAP)\nA: Routine exam\nP: Follow-up appointment in 6 months.\nMedications\nCurrent: Metformin\nAllergies\nLatex\nProblems\nActive Diagnosis: Routine exam\n'
def make_samples(tmp_path):
 d=tmp_path/'samples'; d.mkdir(); shutil.copy2(HAND,d/'SYN-200849_handwritten.png')
 if PDF.exists(): shutil.copy2(PDF,d/'SYN-200989.pdf')
 return d
def make_system(tmp_path):
 provider=DeterministicHashEmbeddingProvider(128)
 store=ChromaStore(embedding_provider=provider,collection_name='day6_documents',persistence_directory=tmp_path/'vectors',backend='local')
 registry=DocumentRegistry(tmp_path/'registry.sqlite3')
 ing=LocalIngestionPipeline(chroma_store=store,registry=registry,config=IngestionConfig(max_characters=300,overlap_characters=40),ocr_function=fake_ocr)
 return ing,store,VectorRetriever(store)
def test_filter_builder_no_filters(): assert build_chroma_where(SearchFilters()) is None
def test_filter_builder_combines_patient_and_page():
 assert build_chroma_where(SearchFilters(patient_id='SYN-200849',page_number=1))=={'$and':[{'patient_id':'SYN-200849'},{'page_number':1}]}
def test_filter_builder_date_range():
 assert build_chroma_where(SearchFilters(date_from=date(2026,1,1),date_to=date(2026,12,31)))=={'$and':[{'document_date':{'$gte':'2026-01-01'}},{'document_date':{'$lte':'2026-12-31'}}]}
def test_unfiltered_retrieval_returns_ranked_passages(tmp_path):
 samples=make_samples(tmp_path); ing,store,r=make_system(tmp_path); assert ing.ingest(samples).failed_files==0
 response=r.search(VectorSearchRequest(query='What allergies are documented?',top_k=5)); assert response.results; assert response.results[0].rank==1; assert all(x.rank==i for i,x in enumerate(response.results,1)); assert all(-1<=x.similarity<=1 for x in response.results)
def test_patient_filter_excludes_other_patient(tmp_path):
 samples=make_samples(tmp_path); ing,store,r=make_system(tmp_path); ing.ingest(samples)
 response=r.search(VectorSearchRequest(query='What allergies are documented?',top_k=10,filters=SearchFilters(patient_id='SYN-200849')))
 assert response.results; assert all(x.patient_id=='SYN-200849' for x in response.results); assert all('Shellfish' not in x.text for x in response.results)
def test_nonexistent_patient_returns_empty(tmp_path):
 samples=make_samples(tmp_path); ing,store,r=make_system(tmp_path); ing.ingest(samples)
 response=r.search(VectorSearchRequest(query='What medication is documented?',top_k=5,filters=SearchFilters(patient_id='MISSING')))
 assert response.results==(); assert response.diagnostics.returned_count==0
def test_filename_filter(tmp_path):
 samples=make_samples(tmp_path); ing,store,r=make_system(tmp_path); ing.ingest(samples)
 response=r.search(VectorSearchRequest(query='primary diagnosis',top_k=10,filters=SearchFilters(filename='SYN-200849_handwritten.png')))
 assert response.results; assert all(x.filename=='SYN-200849_handwritten.png' for x in response.results)
def test_citation_label(tmp_path):
 samples=make_samples(tmp_path); ing,store,r=make_system(tmp_path); ing.ingest(samples)
 response=r.search(VectorSearchRequest(query='medication',top_k=1)); assert 'page 1' in response.results[0].citation_label()
def test_similarity_conversion():
 assert cosine_distance_to_similarity(0.0)==1.0; assert cosine_distance_to_similarity(1.0)==0.0; assert cosine_distance_to_similarity(2.0)==-1.0
