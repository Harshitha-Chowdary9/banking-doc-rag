# Study guide - banking-doc-rag

Run it before any interview (15 minutes):

1. `python -m rag.cli "What should I do if my debit card is stolen?"`
2. `python -m rag.cli --eval` and read the per-question output.
3. `python -m pytest -v` - the test names are the specification.

## Questions you should be able to answer

- **What is RAG and why does an enterprise use it?** LLMs hallucinate and do
  not know private documents. Retrieval-augmented generation first fetches
  the relevant document chunks, then has the model answer using only those
  excerpts - so answers are grounded, citable, and updatable by changing
  documents instead of retraining a model.
- **Why TF-IDF here instead of embeddings?** It is transparent, has zero
  dependencies, runs anywhere, and is genuinely strong on keyword-heavy
  queries over a small corpus. The interface (`search(query, k)`) is the
  same shape a vector store would have, so swapping in embeddings later is a
  localised change. Be honest that semantic paraphrases ("plastic got
  nicked") are where embeddings win over lexical TF-IDF.
- **What did stopword filtering fix?** Words like "a / for / what" appear in
  every chunk; without filtering they let irrelevant chunks beat the document
  that actually contains the topical term. The eval went 11/12 -> 12/12.
- **Why citations?** In a bank, an unsourced answer is not trustworthy.
  Every answer carries the doc ids it was grounded in, and the pipeline
  declines questions the corpus does not cover.
- **How does this connect to the LLM gateway project?** `LLMSummarizer`
  accepts any provider with a `chat(messages)` method - the exact interface
  `enterprise-llm-gateway` exposes - so the gateway's retries, caching and
  offline mock drop straight in front of this pipeline.

## Skills this project genuinely demonstrates

- Python 3 (dataclasses, pathlib, re, json, CLI design, no third-party
  runtime dependencies).
- Information retrieval fundamentals implemented from scratch: TF-IDF,
  IDF smoothing, cosine similarity, stopword filtering, evaluation with a
  labelled question set (top-1/top-3 accuracy).
- The RAG pattern: chunking with provenance, retrieve-then-answer,
  grounded prompts, citations, honest refusal when out of corpus.
- Testing with pytest: unit tests for the maths, corpus loading, pipeline
  behaviour, and the LLM interface with a fake provider.
