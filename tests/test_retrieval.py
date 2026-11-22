from types import SimpleNamespace

import numpy as np
import pytest

from policy_assistant.ingest import build_index
from policy_assistant.models import Chunk
from policy_assistant.retrieval import Retriever, cosine_top_k, openai_query_embedder


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


def _chunks(*section_ids):
    return [
        Chunk(doc_id="d", section_id=s, title=s, text=s, lang="en", version="v")
        for s in section_ids
    ]


def test_retriever_search_returns_chunks_best_first():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    query = np.array([1.0, 0.1])
    retriever = Retriever(_chunks("1.1", "1.2", "1.3"), embeddings, lambda q: query)
    results = retriever.search("q", k=2)
    assert [c.citation_id for c, _ in results] == ["d#1.1", "d#1.3"]
    assert results[0][1] > results[1][1]


def test_retriever_rejects_mismatched_index():
    with pytest.raises(ValueError, match="2 chunks but 1 embedding rows"):
        Retriever(_chunks("1.1", "1.2"), np.array([[1.0, 0.0]]), lambda q: q)


def test_load_missing_index_tells_you_to_run_ingest(tmp_path):
    with pytest.raises(FileNotFoundError, match="policy_assistant.ingest"):
        Retriever.load(tmp_path, lambda q: q)


def _fake_client(vectors_by_text):
    def create(model, input):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=vectors_by_text[t])
                for i, t in enumerate(input)
            ]
        )

    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def test_load_reads_what_build_index_writes(tmp_path):
    chunks = _chunks("1.1", "1.2")
    client = _fake_client({"1.1\n1.1": [1.0, 0.0], "1.2\n1.2": [0.0, 1.0], "q": [0.1, 1.0]})
    build_index(chunks, tmp_path, client, "m")

    retriever = Retriever.load(tmp_path, openai_query_embedder(client, "m"))
    (top, _), *_ = retriever.search("q", k=1)
    assert top == chunks[1]
