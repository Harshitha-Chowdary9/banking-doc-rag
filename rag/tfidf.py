"""A small pure-Python TF-IDF retriever (no third-party dependencies).

Implements the classic bag-of-words TF-IDF weighting with cosine similarity,
the same maths that underpins lexical retrieval in larger RAG systems.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English stopwords carry no topical signal and would otherwise let
# chunks stuffed with "a / for / do" outrank the document that actually
# mentions the query's subject.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "of", "on", "or", "the",
    "to", "what", "when", "where", "which", "who", "why", "with", "you",
    "your", "me", "my", "we", "our", "they", "their", "this", "that",
    "can", "could", "should", "would", "will", "much", "many",
})


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer; punctuation and stopwords dropped."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class TfidfIndex:
    """In-memory TF-IDF index over a list of text chunks."""

    def __init__(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        self.chunks = list(chunks)
        self.n_docs = len(chunks)
        term_freqs: list[Counter] = [Counter(tokenize(c)) for c in chunks]
        self.doc_freqs: Counter = Counter()
        for tf in term_freqs:
            for term in tf:
                self.doc_freqs[term] += 1
        self.vectors = [self._vectorize(tf) for tf in term_freqs]
        self.norms = [math.sqrt(sum(w * w for w in v.values())) or 1.0 for v in self.vectors]

    def idf(self, term: str) -> float:
        """Smoothed inverse document frequency (sklearn-style)."""
        df = self.doc_freqs.get(term, 0)
        return math.log((1 + self.n_docs) / (1 + df)) + 1.0

    def _vectorize(self, tf: Counter) -> dict[str, float]:
        return {term: count * self.idf(term) for term, count in tf.items()}

    def query_vector(self, text: str) -> dict[str, float]:
        return self._vectorize(Counter(tokenize(text)))

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """Return (chunk_index, cosine_score) for the top-k chunks."""
        if k < 1:
            raise ValueError("k must be >= 1")
        qv = self.query_vector(query)
        qnorm = math.sqrt(sum(w * w for w in qv.values()))
        if qnorm == 0.0:
            return []
        scored = []
        for idx, (dv, dnorm) in enumerate(zip(self.vectors, self.norms)):
            dot = sum(w * dv.get(term, 0.0) for term, w in qv.items())
            score = dot / (qnorm * dnorm)
            if score > 0.0:
                scored.append((idx, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
