from pathlib import Path

import numpy as np

from policy_assistant.models import Chunk


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return the k rows of matrix most similar to query, as (row_index, score), best first.

    query has shape (d,), matrix has shape (n, d). If k > n, return all n rows.
    """
    raise NotImplementedError("TODO day 1")


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
