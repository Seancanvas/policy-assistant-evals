import json
from types import SimpleNamespace

import numpy as np

from policy_assistant.ingest import build_index
from policy_assistant.models import Chunk


class FakeClient:
    """Stands in for OpenAI(): records every input it is asked to embed."""

    def __init__(self, dims: int = 3, reverse: bool = False):
        self.calls: list[list[str]] = []
        self.dims = dims
        self.reverse = reverse
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model: str, input: list[str]):
        self.calls.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text))] * self.dims)
            for i, text in enumerate(input)
        ]
        if self.reverse:
            data.reverse()
        return SimpleNamespace(data=data)


def _chunk(section_id: str, text: str) -> Chunk:
    return Chunk(doc_id="d", section_id=section_id, title="T", text=text, lang="zh", version="v")


def test_writes_aligned_files(tmp_path):
    chunks = [_chunk("1.1", "短"), _chunk("1.2", "長一點的內文")]
    build_index(chunks, tmp_path, FakeClient(), "m")

    matrix = np.load(tmp_path / "embeddings.npy")
    records = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    assert [r["section_id"] for r in records] == ["1.1", "1.2"]
    assert records[1]["text"] == "長一點的內文"  # stored as UTF-8, not \u escapes


def test_rerun_with_no_changes_calls_api_zero_times(tmp_path):
    chunks = [_chunk("1.1", "a"), _chunk("1.2", "b")]
    build_index(chunks, tmp_path, FakeClient(), "m")

    client = FakeClient()
    assert build_index(chunks, tmp_path, client, "m") == 0
    assert client.calls == []


def test_only_changed_chunk_is_reembedded(tmp_path):
    build_index([_chunk("1.1", "a"), _chunk("1.2", "b")], tmp_path, FakeClient(), "m")

    client = FakeClient()
    build_index([_chunk("1.1", "a"), _chunk("1.2", "b changed")], tmp_path, client, "m")
    assert client.calls == [["T\nb changed"]]


def test_changing_model_reembeds_everything(tmp_path):
    chunks = [_chunk("1.1", "a"), _chunk("1.2", "b")]
    build_index(chunks, tmp_path, FakeClient(), "model-a")

    client = FakeClient()
    assert build_index(chunks, tmp_path, client, "model-b") == 2


def test_out_of_order_response_is_realigned(tmp_path):
    chunks = [_chunk("1.1", "x"), _chunk("1.2", "xxxxx")]
    build_index(chunks, tmp_path, FakeClient(reverse=True), "m")

    matrix = np.load(tmp_path / "embeddings.npy")
    # The fake vector encodes the text length, so row order is checkable.
    assert matrix[0][0] < matrix[1][0]
