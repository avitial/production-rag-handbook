"""Metadata-filtered vector retrieval.

Pseudo-code:
    validate request
    build metadata filter
    query the vector store
    flatten nested backend response
    convert distance to similarity
    return typed passages with provenance
"""
from __future__ import annotations
from typing import Any
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.retrieval.filter_builder import build_chroma_where
from app.retrieval.models import RetrievedPassage,RetrievalDiagnostics,VectorSearchRequest,VectorSearchResponse
from app.storage.chroma_store import ChromaStore

def cosine_distance_to_similarity(distance: float)->float:
    return max(-1.0,min(1.0,1.0-float(distance)))

def _first(raw: dict[str,Any], key: str)->list[Any]:
    value=raw.get(key,[])
    if not value: return []
    first=value[0]
    return list(first) if first is not None else []

class VectorRetriever:
    def __init__(self, store: ChromaStore)->None:
        self.store=store
    def search(self, request: VectorSearchRequest)->VectorSearchResponse:
        where=build_chroma_where(request.filters)
        count=self.store.count()
        if count==0:
            return VectorSearchResponse(request.query,request.filters,(),RetrievalDiagnostics(0,request.top_k,0,where,self.store.embedding_provider.model_name))
        raw=self.store.query(request.query,top_k=min(request.top_k,count),where=where)
        ids=_first(raw,'ids'); docs=_first(raw,'documents'); metas=_first(raw,'metadatas'); distances=_first(raw,'distances')
        results=[]
        for rank,(chunk_id,text,metadata,distance) in enumerate(zip(ids,docs,metas,distances),start=1):
            metadata=dict(metadata or {})
            results.append(RetrievedPassage(
              chunk_id=str(chunk_id), document_id=str(metadata.get('document_id','')), filename=str(metadata.get('filename','')),
              source_path=str(metadata.get('source_path','')), source_format=str(metadata.get('source_format','')),
              page_number=int(metadata.get('page_number',1)), section=(str(metadata['section']) if metadata.get('section') else None),
              patient_id=(str(metadata['patient_id']) if metadata.get('patient_id') else None), text=str(text), rank=rank,
              distance=float(distance), similarity=cosine_distance_to_similarity(float(distance)), metadata=metadata))
        return VectorSearchResponse(request.query,request.filters,tuple(results),RetrievalDiagnostics(count,request.top_k,len(results),where,self.store.embedding_provider.model_name))
