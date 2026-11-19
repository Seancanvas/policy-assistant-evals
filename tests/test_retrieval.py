import numpy as np

from policy_assistant.models import Chunk
from policy_assistant.retrieval import Retriever, cosine_top_k


def test_cosine_top_k_orders_by_similarity():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    result = cosine_top_k(np.array([1.0, 0.1]), matrix, k=2)
    assert [i for i, _ in result] == [0, 2]


def test_cosine_top_k_caps_k():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert len(cosine_top_k(np.array([1.0, 0.0]), matrix, k=10)) == 2


def test_retriever_search_uses_query_embedding():
    chunks = [
        Chunk(doc_id="d", section_id=s, title=s, text=s, lang="en", version="v")
        for s in ("1.1", "1.2")
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    retriever = Retriever(chunks, embeddings, embed_query=lambda q: np.array([0.0, 1.0]))
    (top, _score), *_ = retriever.search("anything", k=1)
    assert top.citation_id == "d#1.2"
