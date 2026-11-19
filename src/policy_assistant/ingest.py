"""Load the Markdown policy documents, split them into clauses, and embed them.

Run: uv run python -m policy_assistant.ingest
"""

from pathlib import Path

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
    raise NotImplementedError("TODO day 1")


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """Read every *.md file in docs_dir (sorted, so the order is stable) and split it."""
    raise NotImplementedError("TODO day 1")


def build_index(chunks: list[Chunk], index_dir: Path, client, embedding_model: str) -> None:
    """Embed every chunk and write index_dir/chunks.json and index_dir/embeddings.npy.

    TODO day 1:
    - Embed title + text, in batches (client.embeddings.create(model=..., input=[...])).
    - Cache: store a hash of each chunk's content and skip re-embedding unchanged chunks,
      so re-running ingest does not cost money.
    """
    raise NotImplementedError("TODO day 1")


def main() -> None:
    raise NotImplementedError("TODO day 1: load settings, load_chunks, build_index")


if __name__ == "__main__":
    main()
