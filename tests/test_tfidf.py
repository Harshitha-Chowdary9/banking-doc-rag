import math

import pytest

from rag.tfidf import TfidfIndex, tokenize


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("SEPA Transfers, 24/7!") == ["sepa", "transfers", "24", "7"]

    def test_stopwords_removed(self):
        assert tokenize("What is the fee for a transfer?") == ["fee", "transfer"]

    def test_empty(self):
        assert tokenize("!!!") == []


class TestTfidfIndex:
    def test_rejects_empty_corpus(self):
        with pytest.raises(ValueError):
            TfidfIndex([])

    def test_idf_rare_term_beats_common_term(self):
        index = TfidfIndex(["apple banana", "apple cherry", "apple date"])
        assert index.idf("banana") > index.idf("apple")

    def test_unseen_term_idf_is_finite(self):
        index = TfidfIndex(["apple"])
        assert math.isfinite(index.idf("zebra"))

    def test_search_ranks_most_relevant_first(self):
        index = TfidfIndex([
            "overdraft interest rate charges",
            "mortgage fixed rate term",
            "overdraft limit credit check",
        ])
        hits = index.search("overdraft limit", k=2)
        assert hits[0][0] == 2  # both query terms present
        assert hits[0][1] > hits[1][1]

    def test_scores_are_cosine_bounded(self):
        index = TfidfIndex(["overdraft interest", "mortgage rate"])
        for _, score in index.search("overdraft", k=2):
            assert 0.0 <= score <= 1.0 + 1e-9

    def test_k_must_be_positive(self):
        index = TfidfIndex(["text"])
        with pytest.raises(ValueError):
            index.search("text", k=0)

    def test_query_with_no_known_terms_returns_nothing(self):
        index = TfidfIndex(["overdraft policy"])
        assert index.search("zebra quokka", k=3) == []
