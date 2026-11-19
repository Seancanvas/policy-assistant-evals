import pytest

from policy_assistant.ingest import load_chunks, parse_front_matter, split_sections


def _doc(doc_id: str, *section_ids: str) -> str:
    sections = "".join(f"## {s} Title {s}\nText {s}\n" for s in section_ids)
    return f"---\ndoc_id: {doc_id}\nlang: en\nversion: v\n---\n{sections}"


def test_parse_front_matter(sample_doc):
    meta, body = parse_front_matter(sample_doc)
    assert meta["doc_id"] == "sample-doc"
    assert meta["lang"] == "en"
    assert "## 1.1" in body
    assert body.startswith("# Sample policy")
    assert "doc_id" not in body


def test_missing_front_matter_raises():
    with pytest.raises(ValueError):
        parse_front_matter("## 1.1 No front matter\ntext")


def test_unclosed_front_matter_raises():
    with pytest.raises(ValueError, match="closing"):
        parse_front_matter("---\ndoc_id: x\n## 1.1 Clause\ntext")


def test_front_matter_tolerates_extra_spaces():
    meta, _ = parse_front_matter("---\nlang:  en \n\nversion:2026-01\n---\nbody")
    assert meta == {"lang": "en", "version": "2026-01"}


def test_split_sections_one_chunk_per_clause(sample_doc):
    chunks = split_sections(sample_doc)
    assert [c.citation_id for c in chunks] == ["sample-doc#1.1", "sample-doc#1.2"]
    assert chunks[0].title == "Waiting period"
    assert "30 days" in chunks[0].text
    assert "Intro text" not in chunks[0].text
    assert chunks[0].version == "2026-01"
    assert chunks[1].text == "Claims must be filed within 2 years."


def test_split_sections_keeps_subheadings_inside_clause():
    doc = "---\ndoc_id: d\nlang: en\nversion: v\n---\n## 1.1 Clause\nline\n### Note\nmore"
    (chunk,) = split_sections(doc)
    assert chunk.text == "line\n### Note\nmore"


def test_split_sections_heading_without_title_raises():
    with pytest.raises(ValueError, match="Bad heading"):
        split_sections("---\ndoc_id: d\nlang: en\nversion: v\n---\n## 1.1\ntext")


def test_load_chunks_reads_all_docs_in_sorted_order(tmp_path):
    (tmp_path / "b.md").write_text(_doc("b", "1.1"), encoding="utf-8")
    (tmp_path / "a.md").write_text(_doc("a", "1.1", "1.2"), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a doc", encoding="utf-8")
    chunks = load_chunks(tmp_path)
    assert [c.citation_id for c in chunks] == ["a#1.1", "a#1.2", "b#1.1"]


def test_load_chunks_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No .md files"):
        load_chunks(tmp_path)


def test_load_chunks_duplicate_citation_raises(tmp_path):
    (tmp_path / "a.md").write_text(_doc("same", "1.1"), encoding="utf-8")
    (tmp_path / "b.md").write_text(_doc("same", "1.1"), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate citation_id 'same#1.1'"):
        load_chunks(tmp_path)


def test_load_chunks_error_names_the_file(tmp_path):
    (tmp_path / "broken.md").write_text("## 1.1 No front matter\ntext", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.md"):
        load_chunks(tmp_path)
