"""
Runs your actual RAG pipeline (ChromaDB retrieval + Groq generation via
retrival.generate) against every question in eval_dataset.py, and saves the
results in the exact shape RAGAS needs: user_input, retrieved_contexts,
response, reference.

This is a SEPARATE step from actually scoring with RAGAS (run_ragas_eval.py)
for the same reason ingest.py and load_to_chroma.py are separate: you want
to inspect what your pipeline actually retrieved/generated BEFORE spending
LLM-judge calls scoring it. If retrieval looks obviously broken for a
question, you'll see it here before wasting a RAGAS run on it.

Usage:
    python build_ragas_dataset.py --out ./eval_output/ragas_dataset.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAGAS_SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RAGAS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RAGAS_SCRIPT_DIR))

import retrival  # your Groq generation module (generate())
import load_to_chroma
from eval_dataset import EVAL_QUESTIONS

CHROMA_DIR = str(PROJECT_ROOT / "dataset" / "chroma_db")
COLLECTION_NAME = "anatomy_book"


def build_prompt(question: str, context_docs: list[str]) -> str:
    context = "\n\n".join(f"Context chunk:\n{doc}" for doc in context_docs)
    return (
        "You are answering an anatomy question using only the context below.\n\n"
        "Return a complete answer in plain prose. Use the retrieved context only.\n"
        "Do not emit markdown headings such as 'Why', 'Cause', or 'Citations:' inside the answer.\n"
        "Do not stop at an unfinished sentence or fragment.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and use the provided context to support the answer."
    )


def retrieve(collection, question: str, k: int = 3) -> tuple[list[str], list[dict]]:
    """Query Chroma for the top-k chunks. Returns raw chunk texts (for RAGAS's
    retrieved_contexts) and their metadata (for your own inspection/debugging,
    not passed to RAGAS)."""
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return docs, metadatas


def run(out_path: str, k: int = 3):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=load_to_chroma.EMBEDDING_MODEL
    )
    collection = client.get_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    records = []
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        question = item["question"]
        reference = item["reference"]

        docs, metadatas = retrieve(collection, question, k=k)
        prompt = build_prompt(question, docs)
        answer = retrival.generate(prompt, max_new_tokens=512)

        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question}")
        for m in metadatas:
            print(f"    retrieved: {m.get('chapter')} > {m.get('subheading')} (p.{m.get('page_start')})")

        records.append({
            "user_input": question,
            "retrieved_contexts": docs,   # RAGAS needs the raw chunk text, NOT
                                           # your citation-formatted string --
                                           # it needs to independently judge
                                           # whether the response's claims are
                                           # actually supported by this text.
            "response": answer,
            "reference": reference,
        })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./eval_output/ragas_dataset.jsonl")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve per question")
    args = parser.parse_args()
    run(args.out, args.k)