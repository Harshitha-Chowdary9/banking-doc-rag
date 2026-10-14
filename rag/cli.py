"""Command line interface: ask questions, run the retrieval evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .corpus import chunk_documents, load_documents
from .pipeline import RagPipeline

DOCS_DIR = Path(__file__).resolve().parent.parent / "banking_docs"
EVAL_FILE = Path(__file__).resolve().parent.parent / "eval_questions.json"


def run_eval(pipeline: RagPipeline) -> dict:
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    top1 = top3 = 0
    details = []
    for case in cases:
        hits = pipeline.retrieve(case["question"], k=3)
        ranked_doc_ids = [h.chunk.doc_id for h in hits]
        expected = case["expected_doc"]
        top1 += bool(ranked_doc_ids and ranked_doc_ids[0] == expected)
        top3 += expected in ranked_doc_ids
        details.append({
            "question": case["question"],
            "expected": expected,
            "retrieved": ranked_doc_ids,
            "top1_correct": bool(ranked_doc_ids and ranked_doc_ids[0] == expected),
        })
    total = len(cases)
    return {
        "total": total,
        "top1_correct": top1,
        "top1_accuracy": round(top1 / total, 4),
        "top3_correct": top3,
        "top3_accuracy": round(top3 / total, 4),
        "details": details,
    }


def main(argv: list[str]) -> int:
    pipeline = RagPipeline(chunk_documents(load_documents(DOCS_DIR)))
    if len(argv) >= 2 and argv[1] == "--eval":
        report = run_eval(pipeline)
        print(json.dumps(report, indent=2))
        return 0
    if len(argv) < 2:
        print('usage: python -m rag.cli "your question" | --eval')
        return 2
    answer = pipeline.answer(argv[1])
    print(f"Q: {answer.question}")
    print(f"A: {answer.text}")
    print(f"Sources: {', '.join(answer.citations) or 'none'} (mode: {answer.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
