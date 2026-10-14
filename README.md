# banking-doc-rag

Retrieval and summarisation over a set of **fictional** banking documents -
a small, fully offline version of the retrieval-augmented generation (RAG)
pattern used to ground LLM answers in enterprise documents.

> **Learning project.** Built by Harshitha Chittineni with AI-assistant
> pairing as a hands-on study project for LLM/RAG integration patterns. All
> documents describe "Meridian Bank", a fictional institution invented for
> this project. No real bank data, no production use.

## What it does

- Loads 8 fictional banking documents (account terms, overdrafts, SEPA
  transfers, fraud, savings, mortgages, KYC, complaints) and chunks them
  into paragraphs with provenance.
- Indexes the chunks with **TF-IDF written from scratch in pure Python**
  (tokenization, smoothed IDF, cosine similarity, stopword filtering) - no
  scikit-learn or other ML dependencies.
- Answers questions in two interchangeable modes behind one interface:
  - **extractive** (default, offline, deterministic): scores sentences by
    TF-IDF overlap with the question and returns the best, with citations;
  - **LLM mode**: sends the retrieved chunks plus the question to any
    provider implementing a small `chat()` interface (e.g. the mock or
    OpenAI-compatible providers from the sibling `enterprise-llm-gateway`
    project), so an LLM only ever sees grounded context.
- Answers honestly when the corpus does not cover a question instead of
  inventing an answer.
- Ships a 12-question evaluation set with expected source documents.

## Run it (offline, no keys needed)

```bash
python -m rag.cli "What is the monthly fee for the current account?"
python -m rag.cli --eval
```

## Tests and measured results

```bash
pip install -r requirements.txt
python -m pytest
```

**21 tests, all passing** (run 2026-10-14, Python 3.10, pytest 9.1).

Retrieval evaluation on the bundled 12-question set (`python -m rag.cli
--eval`, run 2026-10-14): **top-1 document accuracy 12/12 (100%)**, top-3
12/12. Context matters: this is a tiny 8-document corpus and a hand-written
question set, so the number shows the pipeline works end-to-end - it is not
a benchmark claim. Earlier in development, before stopword filtering, the
same evaluation scored 11/12; the failure mode (common words outranking the
topical term) is fixed and covered by tests.

## What this project deliberately does not claim

- No vector database, no embeddings model, no cloud services, no Docker
  runtime claims - everything here is local pure Python.
- The LLM mode is wired and tested with a fake provider; it was not run
  against a paid live LLM API.
- The corpus is fictional and tiny by design.

See `STUDY_GUIDE.md` for interview talking points and the exact skills this
demonstrates.
