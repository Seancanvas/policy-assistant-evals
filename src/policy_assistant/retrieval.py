from pathlib import Path

import numpy as np

from policy_assistant.models import Chunk


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return the k rows of matrix most similar to query, as (row_index, score), best first.

    query has shape (d,), matrix has shape (n, d). If k > n, return all n rows.
    A zero row scores 0.0; a zero query has no direction, so it raises ValueError.
    """
    if query.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError(f"Shape mismatch: query {query.shape}, matrix {matrix.shape}")
    if k <= 0:
        return []

    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("Query vector is all zeros")
    row_norms = np.linalg.norm(matrix, axis=1)

    dots = matrix @ query
    scores = np.divide(
        dots, row_norms * query_norm, out=np.zeros_like(dots, dtype=float), where=row_norms > 0
    )

    top = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in top]


class Retriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embed_query):
        """embed_query: a function str -> np.ndarray, so tests can pass a fake one."""
        self.chunks = chunks
        self.embeddings = embeddings
        self.embed_query = embed_query

    @classmethod
    def load(cls, index_dir: Path, embed_query) -> "Retriever":
        raise NotImplementedError("TODO day 1: read chunks.json and embeddings.npy")

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        raise NotImplementedError("TODO day 1")
