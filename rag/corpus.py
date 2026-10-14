"""Loading and chunking the fictional banking document corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    chunk_index: int
    text: str


def load_documents(docs_dir: Path) -> dict[str, str]:
    """Read every .md file into {doc_id: raw_text}."""
    docs = {}
    for path in sorted(Path(docs_dir).glob("*.md")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"no .md documents found in {docs_dir}")
    return docs


def chunk_documents(docs: dict[str, str]) -> list[Chunk]:
    """Split documents into paragraph chunks, keeping title + provenance."""
    chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        lines = text.splitlines()
        title = next(
            (ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")),
            doc_id,
        )
        paragraphs = [
            re.sub(r"\s+", " ", p).strip()
            for p in text.split("\n\n")
            if p.strip()
        ]
        body = [p for p in paragraphs if not p.startswith("#")]
        for i, paragraph in enumerate(body):
            chunks.append(Chunk(doc_id=doc_id, title=title, chunk_index=i, text=paragraph))
    return chunks
