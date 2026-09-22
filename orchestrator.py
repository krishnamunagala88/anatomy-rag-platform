import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import load_to_chroma
import retrival


DATASET_DIR = PROJECT_ROOT / "dataset"
JSONL_PATH = str(DATASET_DIR / "output" / "chunks.jsonl")
CHROMA_DIR = str(DATASET_DIR / "chroma_db")
COLLECTION_NAME = "anatomy_book"
DOCUMENT_ID = "anatomy_bd_chaurasia_4thed"


def build_prompt(question: str, context_docs: list[str]) -> str:
    """Turn retrieved anatomy chunks into a Groq-ready prompt."""
    context = "\n\n".join(
        f"Context chunk:\n{doc}" for doc in context_docs
    )

    return (
        "You are answering an anatomy question using only the context below.\n\n"
        "Return a complete answer in plain prose. Use the retrieved context only.\n"
        "Do not emit markdown headings such as 'Why', 'Cause', or 'Citations:' inside the answer.\n"
        "Do not stop at an unfinished sentence or fragment. If the question asks for definition, cause, and treatment, cover all parts clearly.\n"
        "The source citations will be appended by the orchestrator after the answer is produced.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and use the provided context to support the answer."
    )


def append_citations(answer: str, metadatas: list[dict]) -> str:
    """Append a citation list to the answer using the retrieved source metadata."""
    answer = re.sub(r"\n?\s*Citations:\s*.*", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = answer.strip()

    citations = []
    for meta in metadatas:
        chapter = meta.get("chapter", "")
        subheading = meta.get("subheading", "")
        page_start = meta.get("page_start", "")
        citations.append(f"{chapter} > {subheading} (p. {page_start})")

    if not citations:
        return answer

    citation_text = "\n\nCitations:\n" + "\n".join(f"- {item}" for item in citations)
    return answer.rstrip() + citation_text


def print_retrieved_chunks(context_docs: list[str]) -> None:
    """Print retrieved chunks for inspection alongside the generated answer."""
    print("\n=== Retrieved Chunks ===")
    for idx, doc in enumerate(context_docs, start=1):
        print(f"\n--- Chunk {idx} ---\n{doc}\n")


def run_orchestrator(query: str = "what is circumduction", k: int = 3) -> str:
    """Retrieve relevant chunks and generate an answer.

    The collection must be indexed separately with load_to_chroma.py. Querying
    should not re-scan or re-embed the source JSONL for every UI request.
    """
    # Query the already-indexed collection for the closest text chunks.
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=load_to_chroma.EMBEDDING_MODEL
    )
    collection = client.get_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    query_results = collection.query(query_texts=[query], n_results=k)

    docs = query_results.get("documents", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]

    # 3) Build the generation prompt using retrieved documents
    context_docs = []
    retrieved_metas = []
    for idx, doc in enumerate(docs):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        chapter = meta.get("chapter", "")
        subheading = meta.get("subheading", "")
        page_start = meta.get("page_start", "")
        retrieved_metas.append(meta)
        context_docs.append(
            f"Chapter: {chapter}\nSubheading: {subheading}\nPage start: {page_start}\n\n{doc}"
        )

    # Print the retrieved chunks before model generation for inspection.
    print_retrieved_chunks(context_docs)

    prompt = build_prompt(query, context_docs)

    # 4) Call retrival.generate(prompt)
    # A larger completion budget is needed so the model can finish the anatomy
    # answer cleanly before the orchestrator appends the citations section.
    answer = retrival.generate(prompt, max_new_tokens=1024)
    return append_citations(answer, retrieved_metas)


if __name__ == "__main__":
    answer = run_orchestrator("what is varicose veins why its caused and how can it be cured ", k=3)
    print("\n=== Generated Answer ===")
    print(answer)




