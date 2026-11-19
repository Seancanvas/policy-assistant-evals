import pytest

from policy_assistant.ingest import parse_front_matter, split_sections


def test_parse_front_matter(sample_doc):
    meta, body = parse_front_matter(sample_doc)
    assert meta["doc_id"] == "sample-doc"
    assert meta["lang"] == "en"
    assert "## 1.1" in body


def test_missing_front_matter_raises():
    with pytest.raises(ValueError):
        parse_front_matter("## 1.1 No front matter\ntext")


def test_split_sections_one_chunk_per_clause(sample_doc):
    chunks = split_sections(sample_doc)
    assert [c.citation_id for c in chunks] == ["sample-doc#1.1", "sample-doc#1.2"]
    assert chunks[0].title == "Waiting period"
    assert "30 days" in chunks[0].text
    assert "Intro text" not in chunks[0].text
    assert chunks[0].version == "2026-01"
