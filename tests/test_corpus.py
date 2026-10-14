import pytest

from rag.corpus import chunk_documents, load_documents


class TestLoadDocuments:
    def test_loads_real_corpus(self, tmp_path):
        (tmp_path / "alpha.md").write_text("# Title\n\nBody text.", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert docs == {"alpha": "# Title\n\nBody text."}

    def test_missing_dir_content_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path)


class TestChunking:
    def test_paragraphs_become_chunks_with_provenance(self):
        chunks = chunk_documents({"doc-a": "# My Title\n\nFirst para.\n\nSecond para."})
        assert len(chunks) == 2
        assert chunks[0].title == "My Title"
        assert chunks[0].doc_id == "doc-a"
        assert chunks[1].chunk_index == 1
        assert chunks[1].text == "Second para."

    def test_headers_are_not_chunks(self):
        chunks = chunk_documents({"doc-b": "# Only A Header\n\nReal content."})
        assert all(not c.text.startswith("#") for c in chunks)
        assert len(chunks) == 1
