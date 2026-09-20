"""
Load chunks.jsonl (produced by ingest.py) into a persistent ChromaDB collection.

Uses ChromaDB's default embedding function (all-MiniLM-L6-v2, run locally via
onnxruntime, downloaded automatically on first run) -- no external API key
needed. Swap in OpenAIEmbeddingFunction / CohereEmbeddingFunction / etc. from
chromadb.utils.embedding_functions if you want a stronger embedding model.

Usage:
    python load_to_chroma.py --jsonl ./output/chunks.jsonl --chroma_dir ./chroma_db
"""

import argparse
import hashlib
import json
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_JSONL = PROJECT_ROOT / "dataset" / "output" / "chunks.jsonl"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "dataset" / "chroma_db"


def make_chunk_id(document_id: str, text: str) -> str:
    """Must match ingest.py's make_chunk_id exactly. Duplicated here (not
    imported) so this file stays a standalone, independently-runnable script.

    IMPORTANT: this is recomputed from `text` at LOAD time, not trusted from
    whatever chunk_id happens to be stored in the JSONL. This means if you
    hand-edit a chunk's text directly in chunks.jsonl (e.g. to fix an OCR
    error) without re-running ingest.py, the id still updates correctly here
    -- the stored chunk_id field in the JSONL is effectively ignored/stale,
    and this function is the single source of truth for what a chunk's id
    should be, based on its actual current text.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}_{content_hash}"


def load_chunks(jsonl_path: str) -> list[dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_to_metadata(c: dict) -> dict:
    # Chroma wants flat, JSON-serializable metadata values (no lists/dicts).
    # `images` is a list, so we store it as a JSON string and parse it back
    # out at query time -- this is the field we reserved for the later
    # image-linking stage.
    return {
        "chapter": c["chapter"],
        "subheading": c["subheading"],
        "page_start": c["page_start"],
        "page_end": c["page_end"],
        "token_count": c["token_count"],
        "images_json": json.dumps(c["images"]),
    }


def load_to_chroma(jsonl_path: str, chroma_dir: str, collection_name: str = "anatomy_book",
                    document_id: str = "anatomy_bd_chaurasia_4thed"):
    """Incremental load: chunk_id is RECOMPUTED here from each chunk's current
    TEXT (not trusted from the JSONL's stored chunk_id field), so the same
    content always gets the same id across runs -- whether the JSONL came
    from a fresh ingest.py run OR was hand-edited afterward. We use that to
    do three separate things instead of a blind wipe-and-reload:

      1. NEW ids (in this run, not in the DB)         -> add + embed
      2. UNCHANGED ids (in both, same content)         -> refresh metadata
         only (page numbers etc.) -- NO re-embedding, since the text (and
         therefore the correct embedding) hasn't changed
      3. REMOVED ids (in the DB, not in this run)      -> delete -- this
         is content that no longer exists in the source PDF
    """
    chunks = load_chunks(jsonl_path)
    print(f"Loaded {len(chunks)} chunks from {jsonl_path}")

    # Recompute chunk_id from each chunk's CURRENT text -- ignore/overwrite
    # whatever chunk_id is already stored in the JSONL. This is what makes
    # hand-edits to chunks.jsonl (fixing an OCR error, deleting a bad chunk,
    # etc.) work correctly without needing to re-run ingest.py first.
    stale_id_count = 0
    chunks_by_id = {}
    for c in chunks:
        recomputed_id = make_chunk_id(document_id, c["text"])
        if c.get("chunk_id") and c["chunk_id"] != recomputed_id:
            stale_id_count += 1
        chunks_by_id[recomputed_id] = c
    if stale_id_count:
        print(f"  (recomputed {stale_id_count} chunk_ids that didn't match their stored "
              f"value in the JSONL -- likely hand-edited text)")

    new_ids = set(chunks_by_id.keys())

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.get(include=[])  # ids only, cheap call
    existing_ids = set(existing["ids"])

    to_add = new_ids - existing_ids
    to_update = new_ids & existing_ids
    to_delete = existing_ids - new_ids

    print(f"  {len(to_add)} new chunks to embed, "
          f"{len(to_update)} unchanged chunks (metadata-only refresh), "
          f"{len(to_delete)} removed chunks to delete")

    BATCH = 100

    # 1. New content: full add, triggers embedding.
    add_ids = list(to_add)
    for i in range(0, len(add_ids), BATCH):
        batch_ids = add_ids[i:i + BATCH]
        collection.add(
            ids=batch_ids,
            documents=[chunks_by_id[cid]["text"] for cid in batch_ids],
            metadatas=[chunk_to_metadata(chunks_by_id[cid]) for cid in batch_ids],
        )
        print(f"  embedded {min(i + BATCH, len(add_ids))}/{len(add_ids)} new chunks")

    # 2. Unchanged content: metadata-only update (e.g. refreshed page
    # numbers). Passing metadatas without documents/embeddings means Chroma
    # does NOT recompute the embedding -- this is the actual compute saving.
    update_ids = list(to_update)
    for i in range(0, len(update_ids), BATCH):
        batch_ids = update_ids[i:i + BATCH]
        collection.update(
            ids=batch_ids,
            metadatas=[chunk_to_metadata(chunks_by_id[cid]) for cid in batch_ids],
        )
    if update_ids:
        print(f"  refreshed metadata for {len(update_ids)} unchanged chunks (no re-embedding)")

    # 3. Removed content: delete stale vectors for text no longer in the PDF.
    if to_delete:
        collection.delete(ids=list(to_delete))
        print(f"  deleted {len(to_delete)} stale chunks no longer present in the source")

    print(f"\nDone. Collection '{collection_name}' has {collection.count()} chunks "
          f"persisted at {chroma_dir}")
    return collection


def sanity_query(chroma_dir: str, collection_name: str, query: str, k: int = 3):
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    results = collection.query(query_texts=[query], n_results=k)
    print(f"\nQuery: {query!r}")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"  [dist={dist:.3f}] {meta['chapter']} > {meta['subheading']} (p.{meta['page_start']})")
        print(f"    {doc[:150]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--chroma_dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--collection", default="anatomy_book")
    parser.add_argument("--document_id", default="anatomy_bd_chaurasia_4thed",
                         help="Must match the --document_id you used in ingest.py, since "
                              "chunk_ids are recomputed here as {document_id}_{hash(text)}.")
    parser.add_argument("--test_query", default=None,
                         help="Optional: run a sanity-check similarity query after loading")
    args = parser.parse_args()

    load_to_chroma(args.jsonl, args.chroma_dir, args.collection, args.document_id)

    if args.test_query:
        sanity_query(args.chroma_dir, args.collection, args.test_query)