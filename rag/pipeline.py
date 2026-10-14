"""The retrieval + answering pipeline (the "RAG" shape, offline by default).

retrieve -> build a grounded prompt -> answer with citations.

Two answering modes behind one interface:
  - ExtractiveSummarizer (default, offline, deterministic): scores sentences
    of the top chunks by TF-IDF overlap with the query and returns the best.
  - LLMSummarizer: sends retrieved chunks + question to any LLMProvider, such
    as the mock or OpenAI-compatible providers from the sibling
    enterprise-llm-gateway project. Requires the caller to supply a provider;
    this project itself never needs a key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Chunk
from .tfidf import TfidfIndex, tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    citations: list[str]   # doc_ids the answer is grounded in
    mode: str              # "extractive" or "llm"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


class ExtractiveSummarizer:
    """Offline sentence-scoring summarizer (no model, no network)."""

    def __init__(self, index: TfidfIndex) -> None:
        self.index = index

    def answer(self, question: str, results: list[RetrievalResult]) -> str:
        query_terms = set(tokenize(question))
        best: list[tuple[float, str]] = []
        for result in results[:2]:
            for sentence in split_sentences(result.chunk.text):
                terms = set(tokenize(sentence))
                overlap = query_terms & terms
                score = sum(self.index.idf(t) for t in overlap)
                if score > 0:
                    best.append((score, sentence))
        if not best:
            return "The fictional corpus does not cover this question."
        best.sort(key=lambda pair: pair[0], reverse=True)
        return " ".join(sentence for _, sentence in best[:2])


class LLMSummarizer:
    """Grounds an LLM answer in the retrieved chunks."""

    def __init__(self, provider) -> None:
        self.provider = provider

    def answer(self, question: str, results: list[RetrievalResult]) -> str:
        context = "\n\n".join(
            f"[{r.chunk.doc_id}] {r.chunk.text}" for r in results
        )
        messages = [
            {"role": "system", "content": (
                "Answer the question using only the provided banking document "
                "excerpts. Cite doc ids in brackets."
            )},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"},
        ]
        return self.provider.chat(messages).content


class RagPipeline:
    def __init__(self, chunks: list[Chunk], summarizer=None) -> None:
        self.chunks = chunks
        self.index = TfidfIndex([c.text for c in chunks])
        self.summarizer = summarizer or ExtractiveSummarizer(self.index)

    def retrieve(self, question: str, k: int = 3) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=self.chunks[i], score=score)
            for i, score in self.index.search(question, k=k)
            if score > 0.0
        ]

    def answer(self, question: str, k: int = 3) -> Answer:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        results = self.retrieve(question, k=k)
        if not results:
            return Answer(question=question, text="The fictional corpus does not cover this question.",
                          citations=[], mode="none")
        text = self.summarizer.answer(question, results)
        citations = sorted({r.chunk.doc_id for r in results})
        mode = "llm" if isinstance(self.summarizer, LLMSummarizer) else "extractive"
        return Answer(question=question, text=text, citations=citations, mode=mode)


def build_pipeline(docs_dir) -> "RagPipeline":
    from .corpus import chunk_documents, load_documents
    return RagPipeline(chunk_documents(load_documents(docs_dir)))
