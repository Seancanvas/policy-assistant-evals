import json
from collections.abc import Callable
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


EmbedQuery = Callable[[str], np.ndarray]


def openai_query_embedder(client, embedding_model: str) -> EmbedQuery:
    """Return a str -> vector function backed by the OpenAI embeddings API."""

    def embed(text: str) -> np.ndarray:
        response = client.embeddings.create(model=embedding_model, input=[text])
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    return embed


class Retriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embed_query: EmbedQuery):
        """embed_query: a function str -> np.ndarray, so tests can pass a fake one."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embedding rows")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embed_query = embed_query

    @classmethod
    def load(cls, index_dir: Path, embed_query: EmbedQuery) -> "Retriever":
        chunks_path = index_dir / "chunks.json"
        vectors_path = index_dir / "embeddings.npy"
        if not (chunks_path.exists() and vectors_path.exists()):
            raise FileNotFoundError(
                f"No index in {index_dir}. Run: uv run python -m policy_assistant.ingest"
            )
        records = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**record) for record in records]
        return cls(chunks, np.load(vectors_path), embed_query)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Return the k chunks most similar to query, as (chunk, score), best first."""
        vector = self.embed_query(query)
        hits = cosine_top_k(vector, self.embeddings, k)
        return [(self.chunks[i], score) for i, score in hits]
