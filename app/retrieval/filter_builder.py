"""Build Chroma-compatible metadata filters.

Pseudo-code:
    collect a condition for each populated filter
    return None for no conditions
    return the single condition when one exists
    otherwise return {"$and": conditions}
"""
from __future__ import annotations
from typing import Any
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.retrieval.models import SearchFilters

def _clean(value: str|None, field_name: str)->str|None:
    if value is None: return None
    cleaned=value.strip()
    if not cleaned: raise ValueError(f'{field_name} must not be blank')
    return cleaned

def build_chroma_where(filters: SearchFilters)->dict[str,Any]|None:
    conditions=[]
    values={
      'patient_id':_clean(filters.patient_id,'patient_id'),
      'document_id':_clean(filters.document_id,'document_id'),
      'document_type':_clean(filters.document_type,'document_type'),
      'filename':_clean(filters.filename,'filename'),
      'source_format':_clean(filters.source_format,'source_format'),
      'page_number':filters.page_number,
    }
    for key,value in values.items():
        if value is not None: conditions.append({key:value})
    if filters.date_from is not None:
        conditions.append({'document_date':{'$gte':filters.date_from.isoformat()}})
    if filters.date_to is not None:
        conditions.append({'document_date':{'$lte':filters.date_to.isoformat()}})
    if not conditions: return None
    if len(conditions)==1: return conditions[0]
    return {'$and':conditions}
