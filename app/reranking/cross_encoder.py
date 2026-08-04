"""Cross-encoder reranking with a deterministic offline fallback.

Production pseudo-code:
    pairs=[(query,passage) for passage in passages]
    scores=CrossEncoder.predict(pairs)
    return one relevance score per passage
"""
from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
import re
from typing import Any
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))
from app.reranking.base import PassageReranker

class CrossEncoderReranker(PassageReranker):
    def __init__(self,model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',*,batch_size=16,model:Any|None=None):
        if batch_size<=0: raise ValueError('batch_size must be greater than zero')
        if model is None:
            try: from sentence_transformers import CrossEncoder
            except ImportError as e: raise RuntimeError('install sentence-transformers or use deterministic backend') from e
            model=CrossEncoder(model_name)
        self._model=model; self._model_name=model_name; self._batch_size=batch_size
    @property
    def model_name(self): return self._model_name
    def score_pairs(self,query:str,passages:Sequence[str])->list[float]:
        if not query.strip(): raise ValueError('query must not be blank')
        if not passages: return []
        pairs=[[query,p] for p in passages]
        raw=self._model.predict(pairs,batch_size=self._batch_size,show_progress_bar=False)
        out=[]
        for x in raw:
            try:
                if not isinstance(x,(str,bytes)) and hasattr(x,'__len__'): x=x[0]
            except Exception: pass
            out.append(float(x))
        return out

class DeterministicReranker(PassageReranker):
    """Dependency-free test reranker; not quality-equivalent to a learned model."""
    TOKEN=re.compile(r'[a-z0-9]+(?:[-_/][a-z0-9]+)*',re.I)
    @property
    def model_name(self): return 'deterministic-overlap-reranker-v1'
    def _tokens(self,text): return [m.group(0).lower() for m in self.TOKEN.finditer(text)]
    def _score(self,q,p):
        qt=self._tokens(q); pt=self._tokens(p)
        if not qt or not pt: return 0.0
        qc,pc=Counter(qt),Counter(pt)
        overlap=sum(min(n,pc[t]) for t,n in qc.items())/max(1,len(qt))
        specific=sum(0.35 for t in qc if (any(c.isdigit() for c in t) or '-' in t or '/' in t or len(t)>=8) and t in pc)
        qn=' '.join(qt); pn=' '.join(pt)
        phrase=1.0 if qn and qn in pn else 0.0
        labels=sum(0.25 for label in ('medications','allergies','diagnosis','problems','follow-up','plan','referral') if label in qn and label in pn)
        return overlap+specific+phrase+labels
    def score_pairs(self,query:str,passages:Sequence[str])->list[float]: return [self._score(query,p) for p in passages]

def create_reranker(backend='auto',*,model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')->PassageReranker:
    b=backend.strip().lower()
    if b=='deterministic': return DeterministicReranker()
    if b=='cross-encoder': return CrossEncoderReranker(model_name)
    if b=='auto':
        try: return CrossEncoderReranker(model_name)
        except Exception: return DeterministicReranker()
    raise ValueError('reranker backend must be auto, cross-encoder, or deterministic')
