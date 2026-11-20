import numpy as np
import pytest

from policy_assistant.models import Chunk
from policy_assistant.retrieval import Retriever, cosine_top_k


def test_cosine_top_k_orders_by_similarity():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    result = cosine_top_k(np.array([1.0, 0.1]), matrix, k=2)
    assert [i for i, _ in result] == [0, 2]


def test_cosine_top_k_caps_k():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert len(cosine_top_k(np.array([1.0, 0.0]), matrix, k=10)) == 2


def test_cosine_ignores_vector_length():
    # Same direction, different lengths: both must score 1.0.
    matrix = np.array([[2.0, 0.0], [100.0, 0.0], [0.0, 5.0]])
    result = cosine_top_k(np.array([3.0, 0.0]), matrix, k=3)
    assert [round(s, 6) for _, s in result] == [1.0, 1.0, 0.0]


def test_cosine_returns_python_types_best_first():
    result = cosine_top_k(np.array([1.0, 0.0]), np.array([[0.0, 1.0], [1.0, 1.0]]), k=2)
    assert all(type(i) is int and type(s) is float for i, s in result)
    assert result[0][1] >= result[1][1]


def test_cosine_zero_row_scores_zero():
    result = cosine_top_k(np.array([1.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0]]), k=2)
    assert result == [(1, 1.0), (0, 0.0)]


def test_cosine_zero_query_raises():
    with pytest.raises(ValueError, match="zeros"):
        cosine_top_k(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]), k=1)


def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        cosine_top_k(np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0]]), k=1)


@pytest.mark.parametrize("k", [0, -1])
def test_cosine_non_positive_k_returns_empty(k):
    assert cosine_top_k(np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), k=k) == []


def test_retriever_search_uses_query_embedding():
    chunks = [
        Chunk(doc_id="d", section_id=s, title=s, text=s, lang="en", version="v")
        for s in ("1.1", "1.2")
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    retriever = Retriever(chunks, embeddings, embed_query=lambda q: np.array([0.0, 1.0]))
    (top, _score), *_ = retriever.search("anything", k=1)
    assert top.citation_id == "d#1.2"
