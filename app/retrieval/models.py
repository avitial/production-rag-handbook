"""Typed models for metadata-filtered vector retrieval.

Pseudo-code:
    SearchFilters -> optional constraints
    VectorSearchRequest -> query + top-k + filters
    RetrievedPassage -> ranked chunk with provenance
    VectorSearchResponse -> typed results + diagnostics
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any

@dataclass(frozen=True)
class SearchFilters:
    patient_id: str|None=None
    document_id: str|None=None
    document_type: str|None=None
    filename: str|None=None
    source_format: str|None=None
    page_number: int|None=None
    date_from: date|None=None
    date_to: date|None=None
    def __post_init__(self):
        if self.page_number is not None and self.page_number < 1:
            raise ValueError('page_number must be at least 1')
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError('date_from cannot be after date_to')

@dataclass(frozen=True)
class VectorSearchRequest:
    query: str
    top_k: int=10
    filters: SearchFilters=field(default_factory=SearchFilters)
    def __post_init__(self):
        if not self.query.strip(): raise ValueError('query must not be blank')
        if self.top_k <= 0: raise ValueError('top_k must be greater than zero')

@dataclass(frozen=True)
class RetrievedPassage:
    chunk_id: str
    document_id: str
    filename: str
    source_path: str
    source_format: str
    page_number: int
    section: str|None
    patient_id: str|None
    text: str
    rank: int
    distance: float
    similarity: float
    metadata: dict[str,Any]=field(default_factory=dict)
    def citation_label(self)->str:
        section=f', section {self.section}' if self.section else ''
        return f'{self.filename}, page {self.page_number}{section}'

@dataclass(frozen=True)
class RetrievalDiagnostics:
    collection_count: int
    requested_top_k: int
    returned_count: int
    where_filter: dict[str,Any]|None
    embedding_model: str

@dataclass(frozen=True)
class VectorSearchResponse:
    query: str
    filters: SearchFilters
    results: tuple[RetrievedPassage,...]
    diagnostics: RetrievalDiagnostics
