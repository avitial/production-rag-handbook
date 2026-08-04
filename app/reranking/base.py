"""Shared reranking contracts.

Pseudo-code:
    validate query and candidates
    score each query-passage pair
    sort by reranker score
    preserve original retrieval rank, score, and provenance
    return top_n passages plus timing diagnostics
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Sequence
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.retrieval.models import RetrievedPassage

@dataclass(frozen=True)
class RerankingRequest:
    query: str
    passages: tuple[RetrievedPassage,...]
    top_n: int=5
    def __post_init__(self):
        if not self.query.strip(): raise ValueError('query must not be blank')
        if self.top_n<=0: raise ValueError('top_n must be greater than zero')

@dataclass(frozen=True)
class RerankedPassage:
    passage: RetrievedPassage
    rerank_rank: int
    rerank_score: float
    original_rank: int
    original_similarity: float
    model_name: str
    metadata: dict[str,Any]=field(default_factory=dict)
    def citation_label(self)->str: return self.passage.citation_label()

@dataclass(frozen=True)
class RerankingDiagnostics:
    model_name: str
    candidate_count: int
    returned_count: int
    duration_ms: float

@dataclass(frozen=True)
class RerankingResponse:
    query: str
    results: tuple[RerankedPassage,...]
    diagnostics: RerankingDiagnostics

class PassageReranker(ABC):
    @property
    @abstractmethod
    def model_name(self)->str: ...
    @abstractmethod
    def score_pairs(self,query:str,passages:Sequence[str])->list[float]: ...
    def rerank(self,request:RerankingRequest)->RerankingResponse:
        started=perf_counter()
        if not request.passages:
            return RerankingResponse(request.query,(),RerankingDiagnostics(self.model_name,0,0,(perf_counter()-started)*1000))
        scores=self.score_pairs(request.query,[p.text for p in request.passages])
        if len(scores)!=len(request.passages): raise ValueError('reranker score count does not match candidate count')
        ranked=sorted(zip(request.passages,scores),key=lambda x:(-float(x[1]),x[0].rank,x[0].chunk_id))
        out=[]
        for rank,(p,score) in enumerate(ranked[:request.top_n],1):
            out.append(RerankedPassage(p,rank,float(score),p.rank,p.similarity,self.model_name,{
                'retrieval_method':p.metadata.get('retrieval_method','unknown'),
                'original_distance':p.distance,
            }))
        return RerankingResponse(request.query,tuple(out),RerankingDiagnostics(self.model_name,len(request.passages),len(out),(perf_counter()-started)*1000))
