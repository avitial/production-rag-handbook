"""Unit tests for Day 1 embeddings."""
from __future__ import annotations
import os
import numpy as np
import pytest
import sys

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook/scripts"))
from embedding_demo import cosine_scores, encode_texts, semantic_search

class FakeEmbeddingModel:
    vectors = {
        "hypertension": np.array([1.0,0.0,0.0], dtype=np.float32),
        "blood pressure medicine": np.array([0.9,0.1,0.0], dtype=np.float32),
        "lisinopril treats hypertension": np.array([0.95,0.05,0.0], dtype=np.float32),
        "no known allergies": np.array([0.0,1.0,0.0], dtype=np.float32),
        "cardiology follow-up": np.array([0.0,0.0,1.0], dtype=np.float32),
    }
    def encode(self, texts, *, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        del convert_to_numpy, show_progress_bar
        rows=[]
        for text in texts:
            vector=self.vectors[text].copy()
            if normalize_embeddings:
                norm=np.linalg.norm(vector)
                if norm: vector=vector/norm
            rows.append(vector)
        return np.vstack(rows)

def test_encode_texts_returns_two_dimensional_array():
    embeddings=encode_texts(FakeEmbeddingModel(), ["hypertension","no known allergies"])
    assert embeddings.shape == (2,3)
    assert embeddings.dtype == np.float32

def test_encode_texts_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        encode_texts(FakeEmbeddingModel(), [])

def test_cosine_scores_rank_similar_vector_higher():
    passages=np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32)
    scores=cosine_scores(passages, np.array([1.,0.,0.], dtype=np.float32))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)

def test_cosine_scores_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="same dimensions"):
        cosine_scores(np.ones((2,3), dtype=np.float32), np.ones(4, dtype=np.float32))

def test_semantic_search_places_most_related_passage_first():
    results=semantic_search(FakeEmbeddingModel(), "blood pressure medicine", ["no known allergies","cardiology follow-up","lisinopril treats hypertension"], top_k=2)
    assert len(results)==2
    assert results[0].passage == "lisinopril treats hypertension"
    assert results[0].rank == 1
    assert results[0].score > results[1].score

def test_semantic_search_rejects_blank_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        semantic_search(FakeEmbeddingModel(), "   ", ["hypertension"])

@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="Set RUN_MODEL_TESTS=1 to test the real model.")
def test_real_sentence_transformer_understands_semantic_similarity():
    from embedding_demo import load_embedding_model
    results=semantic_search(load_embedding_model(), "What medicine treats high blood pressure?", ["The patient has no known allergies.","The patient takes lisinopril for hypertension.","The patient has a follow-up appointment."], top_k=1)
    assert "lisinopril" in results[0].passage.lower()
