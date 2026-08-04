"""Append-only JSONL traces for retrieval and reranking.

Pseudo-code:
    align retrieval candidates with reranked results by chunk_id
    retain before/after ranks and scores
    store short excerpts and source metadata
    append one JSON object per line
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json, uuid
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.reranking.base import RerankingResponse
from app.retrieval.models import VectorSearchResponse

@dataclass(frozen=True)
class TraceCandidate:
    chunk_id:str; document_id:str; patient_id:str|None; filename:str; page_number:int; section:str|None
    text_excerpt:str; original_rank:int; original_score:float; rerank_rank:int|None=None; rerank_score:float|None=None

@dataclass(frozen=True)
class RetrievalTrace:
    trace_id:str; timestamp_utc:str; query:str; filters:dict[str,Any]; retrieval_method:str
    retrieval_model:str; reranker_model:str|None; retrieval_candidate_count:int; final_result_count:int
    retrieval_duration_ms:float|None; reranking_duration_ms:float|None; candidates:tuple[TraceCandidate,...]
    metadata:dict[str,Any]=field(default_factory=dict)

def create_retrieval_trace(*,retrieval_response:VectorSearchResponse,reranking_response:RerankingResponse|None=None,retrieval_method='hybrid',retrieval_duration_ms=None,excerpt_characters=240,metadata=None)->RetrievalTrace:
    if excerpt_characters<=0: raise ValueError('excerpt_characters must be greater than zero')
    reranked={}; model=None; rerank_ms=None
    if reranking_response:
        reranked={r.passage.chunk_id:r for r in reranking_response.results}; model=reranking_response.diagnostics.model_name; rerank_ms=reranking_response.diagnostics.duration_ms
    candidates=[]
    for p in retrieval_response.results:
        r=reranked.get(p.chunk_id)
        candidates.append(TraceCandidate(p.chunk_id,p.document_id,p.patient_id,p.filename,p.page_number,p.section,' '.join(p.text.split())[:excerpt_characters],p.rank,p.similarity,r.rerank_rank if r else None,r.rerank_score if r else None))
    filters={k:(v.isoformat() if hasattr(v,'isoformat') else v) for k,v in asdict(retrieval_response.filters).items() if v is not None}
    return RetrievalTrace(str(uuid.uuid4()),datetime.now(timezone.utc).isoformat(),retrieval_response.query,filters,retrieval_method,retrieval_response.diagnostics.embedding_model,model,len(retrieval_response.results),len(reranking_response.results) if reranking_response else len(retrieval_response.results),retrieval_duration_ms,rerank_ms,tuple(candidates),dict(metadata or {}))

class RetrievalTraceWriter:
    def __init__(self,path:str|Path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def write(self,trace:RetrievalTrace)->None:
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(asdict(trace),ensure_ascii=False,sort_keys=True)+'\n'); f.flush()
    def read_all(self)->list[dict[str,Any]]:
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
