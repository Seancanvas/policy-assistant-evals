"""Load the Markdown policy documents, split them into clauses, and embed them.

Run: uv run python -m policy_assistant.ingest
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from policy_assistant.config import DOCS_DIR, INDEX_DIR, load_settings
from policy_assistant.models import Chunk


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a document into its front matter (the block between the two '---' lines) and body.

    Returns (metadata, body). Raise ValueError if the front matter is missing.
    """
    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        raise ValueError("Missing front matter: the document must start with '---'")

    try:
        end = lines.index("---", 1)
    except ValueError:
        raise ValueError("Unclosed front matter: no closing '---' line") from None

    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"Bad front matter line (expected 'key: value'): {line!r}")
        meta[key.strip()] = value.strip()

    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def split_sections(text: str) -> list[Chunk]:
    """Turn one Markdown document into one Chunk per '## <section_id> <title>' heading.

    Text before the first '##' heading is ignored. doc_id, lang and version come from the
    front matter.
    """
    meta, body = parse_front_matter(text)

    chunks: list[Chunk] = []
    heading: tuple[str, str] | None = None
    lines: list[str] = []

    def flush() -> None:
        if heading is not None:
            section_id, title = heading
            chunks.append(
                Chunk(
                    doc_id=meta["doc_id"],
                    section_id=section_id,
                    title=title,
                    text="\n".join(lines).strip(),
                    lang=meta["lang"],
                    version=meta["version"],
                )
            )

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            parts = line[3:].split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad heading (expected '## <section_id> <title>'): {line!r}")
            heading = (parts[0], parts[1].strip())
            lines = []
        elif heading is not None:
            lines.append(line)
    flush()

    return chunks


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """Read every *.md file in docs_dir (sorted, so the order is stable) and split it."""
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"No .md files found in {docs_dir}")

    chunks: list[Chunk] = []
    seen: dict[str, str] = {}
    for path in paths:
        try:
            doc_chunks = split_sections(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        for chunk in doc_chunks:
            if chunk.citation_id in seen:
                raise ValueError(
                    f"Duplicate citation_id {chunk.citation_id!r} "
                    f"in {seen[chunk.citation_id]} and {path.name}"
                )
            seen[chunk.citation_id] = path.name
        chunks.extend(doc_chunks)

    return chunks


BATCH_SIZE = 100


def embedding_text(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.text}"


def content_hash(text: str, embedding_model: str) -> str:
    # The model is part of the key: a vector from another model is not reusable.
    return hashlib.sha256(f"{embedding_model}\n{text}".encode()).hexdigest()


def _load_cache(index_dir: Path) -> dict[str, np.ndarray]:
    chunks_path = index_dir / "chunks.json"
    vectors_path = index_dir / "embeddings.npy"
    if not (chunks_path.exists() and vectors_path.exists()):
        return {}
    records = json.loads(chunks_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)
    if len(records) != len(vectors):
        return {}  # Inconsistent index: rebuild everything rather than trust it.
    return {
        r["content_hash"]: v for r, v in zip(records, vectors, strict=True) if "content_hash" in r
    }


def _embed(client, embedding_model: str, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=embedding_model, input=batch)
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} embeddings, got {len(items)}")
        vectors.extend(item.embedding for item in items)
    return vectors


def build_index(chunks: list[Chunk], index_dir: Path, client, embedding_model: str) -> int:
    """Embed every chunk and write index_dir/chunks.json and index_dir/embeddings.npy.

    Row i of embeddings.npy is the vector for record i of chunks.json. Chunks whose
    content and model are unchanged reuse the cached vector, so re-running costs nothing.
    Returns the number of chunks that were sent to the API.
    """
    if not chunks:
        raise ValueError("No chunks to index")
    index_dir.mkdir(parents=True, exist_ok=True)

    texts = [embedding_text(c) for c in chunks]
    hashes = [content_hash(t, embedding_model) for t in texts]
    cache = _load_cache(index_dir)

    missing = [i for i, h in enumerate(hashes) if h not in cache]
    new_vectors = _embed(client, embedding_model, [texts[i] for i in missing])
    for i, vector in zip(missing, new_vectors, strict=True):
        cache[hashes[i]] = np.asarray(vector, dtype=np.float32)

    matrix = np.stack([cache[h] for h in hashes]).astype(np.float32)
    records = [c.model_dump() | {"content_hash": h} for c, h in zip(chunks, hashes, strict=True)]

    np.save(index_dir / "embeddings.npy", matrix)
    (index_dir / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(missing)


def main() -> None:
    settings = load_settings()
    chunks = load_chunks(DOCS_DIR)
    embedded = build_index(chunks, INDEX_DIR, OpenAI(), settings.embedding_model)
    print(
        f"Indexed {len(chunks)} chunks into {INDEX_DIR} ({embedded} embedded, "
        f"{len(chunks) - embedded} from cache)"
    )


if __name__ == "__main__":
    main()
