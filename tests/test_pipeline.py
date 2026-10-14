import pytest

from rag.corpus import chunk_documents
from rag.pipeline import (
    ExtractiveSummarizer,
    LLMSummarizer,
    RagPipeline,
    split_sentences,
)

DOCS = {
    "overdraft-policy": (
        "# Overdraft Policy\n\n"
        "The arranged overdraft interest rate is 9.9% APR. It is calculated daily "
        "and charged monthly.\n\n"
        "Customers receive an SMS alert at 80% of the arranged limit."
    ),
    "savings-interest": (
        "# Savings\n\n"
        "The Instant Savings account pays 2.10% AER on balances up to 50,000 EUR."
    ),
}


def make_pipeline(summarizer=None):
    return RagPipeline(chunk_documents(DOCS), summarizer=summarizer)


class RecordingProvider:
    def __init__(self):
        self.messages = None

    def chat(self, messages, *, temperature=0.2):
        self.messages = messages
        class Resp:
            content = "grounded answer [overdraft-policy]"
        return Resp()


class TestRetrieve:
    def test_retrieves_relevant_doc(self):
        pipeline = make_pipeline()
        hits = pipeline.retrieve("what is the overdraft interest rate?")
        assert hits[0].chunk.doc_id == "overdraft-policy"
        assert hits[0].score > 0

    def test_irrelevant_question_returns_no_hits(self):
        pipeline = make_pipeline()
        assert pipeline.retrieve("zebra quokka platypus") == []


class TestAnswer:
    def test_extractive_answer_contains_key_fact_and_citation(self):
        pipeline = make_pipeline()
        answer = pipeline.answer("what is the overdraft interest rate?")
        assert "9.9%" in answer.text
        assert "overdraft-policy" in answer.citations
        assert answer.mode == "extractive"

    def test_uncovered_question_is_honest(self):
        pipeline = make_pipeline()
        answer = pipeline.answer("what is the airspeed of a swallow?")
        assert "does not cover" in answer.text
        assert answer.citations == []

    def test_llm_mode_sends_grounded_prompt(self):
        provider = RecordingProvider()
        pipeline = make_pipeline(summarizer=LLMSummarizer(provider))
        answer = pipeline.answer("overdraft rate?")
        assert answer.mode == "llm"
        assert answer.text == "grounded answer [overdraft-policy]"
        system, user = provider.messages
        assert "only the provided" in system["content"]
        assert "[overdraft-policy]" in user["content"]

    def test_blank_question_rejected(self):
        pipeline = make_pipeline()
        with pytest.raises(ValueError):
            pipeline.answer("   ")


class TestSentences:
    def test_split(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
