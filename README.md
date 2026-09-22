# Anatomy RAG

A domain-specific Retrieval-Augmented Generation (RAG) project built for anatomy textbook Q&A. The system ingests textbook content, stores chunked text in ChromaDB, retrieves the most relevant passages for a user question, and generates grounded answers using a Groq LLM while preserving source citations.

This project is designed to answer anatomy questions with citations and context from a textbook, rather than relying on generic web knowledge.

## Why this project matters

Medical and academic knowledge work is highly specialized. Generic LLMs can answer broad questions, but they often miss the precise terminology, structure, and definitions you find in a textbook.

This project demonstrates a practical RAG pattern for a domain-specific corpus:

- ingest raw educational content from a PDF
- split it into consistent, semantically meaningful chunks
- embed and store those chunks in a vector database
- retrieve the most relevant passages for a query
- generate a grounded answer in plain language
- cite the exact textbook sections that support the answer

## Architecture

- `ingest.py` — extracts textbook text and builds structured, heading-aware chunks with content-hash-based IDs (supports incremental re-indexing)
- `load_to_chroma.py` — embeds chunks with a medical domain model and stores them in ChromaDB, using a diff-based upsert (new / unchanged / deleted) rather than a full rebuild each run
- `orchestrator.py` — retrieves relevant chunks and builds the LLM prompt
- `retrival.py` — calls the Groq model to generate answers
- `ui/anatomy_ui.py` — Streamlit interface for interactive Q&A
- `ragas/` — evaluation pipeline for retrieval and generation quality

## Embedding strategy

This project uses a biomedical embedding model instead of a generic sentence embedding model:

- `NeuML/pubmedbert-base-embeddings`

The intuition was that anatomy terminology is specialized, so a biomedical-tuned model should retrieve better than a general-purpose one. **In practice, on this book, it didn't** — see Evaluation Results below.

## Workflow

1. Ingest the textbook PDF into chunked JSONL output.
2. Load chunks into ChromaDB with the medical embedding model.
3. Query the vector store for the most relevant passages.
4. Build a prompt with the retrieved context and the user question.
5. Generate a grounded answer from the textbook passages.
6. Append citations from the metadata retrieved alongside the chunks.

## Tech stack

- Python
- ChromaDB
- SentenceTransformers
- Streamlit
- Groq
- RAGAS
- pdfplumber / pypdf

## Project structure

```text
Anatomy RAG/
├── ingest.py
├── load_to_chroma.py
├── orchestrator.py
├── retrival.py
├── requirements.txt
├── README.md
├── .env.example
├── assets/
│   └── anatomy.pdf
├── dataset/
│   ├── chroma_db/
│   ├── eval_output/
│   └── output/
├── ragas/
│   ├── build_ragas_dataset.py
│   ├── eval_dataset.py
│   ├── ragas_evaluation.py
│   └── run_ragas_eval.py
├── tests/
│   └── test_orchestrator.py
└── ui/
    └── anatomy_ui.py
```

## Setup

1. Create and activate a virtual environment.

```powershell
cd "C:\Users\HP\Desktop\Anatomy RAG"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your Groq API key.

```powershell
copy .env.example .env
```

```text
GROQ_API_KEY=your_key_here
```

## Ingest the textbook

```powershell
python .\ingest.py `
  --pdf .\assets\anatomy.pdf `
  --out_dir .\dataset\output `
  --document_id anatomy_bd_chaurasia_4thed
```

## Index the textbook content

```powershell
python .\load_to_chroma.py `
  --jsonl .\dataset\output\chunks.jsonl `
  --chroma_dir .\dataset\chroma_db
```

## Run the Streamlit UI

```powershell
python -m streamlit run .\ui\anatomy_ui.py
```

## Run RAGAS evaluation

Generate the dataset:

```powershell
python .\ragas\build_ragas_dataset.py --out .\dataset\eval_output\ragas_dataset.jsonl
```

Score the dataset:

```powershell
python .\ragas\run_ragas_eval.py --dataset .\dataset\eval_output\ragas_dataset.jsonl
```

The judge LLM is configurable in `run_ragas_eval.py` — swap in a different Groq model, or a different provider entirely, to reduce same-model judge bias (see Limitations below).

## Example use case

The interface allows a user to ask questions such as:

- What is circumduction?
- What is the anatomical position?
- What is the difference between pronation and supination?
- What is a synovial joint?

The system retrieves textbook passages, builds a grounded answer, and shows the source citations from the original book.

## Evaluation results

Ran the same pipeline with two different embedding models, scored with RAGAS (faithfulness, answer relevancy, context precision, context recall) on a 15-question hand-written eval set:

| Metric | all-MiniLM-L6-v2 (default) | NeuML/pubmedbert-base-embeddings |
|---|---|---|
| faithfulness | ~0.91 | 0.736 |
| answer_relevancy | ~0.94 | 0.925 |
| context_precision_without_reference | ~0.85 | 0.788 |
| context_recall | ~0.87 | 0.769 |

The general-purpose model outperformed the biomedical-domain-specific one on every metric. Working theory: PubMedBERT is fine-tuned on formal PubMed research-abstract prose, while this textbook is written in terse, glossary-style definitions — a different register, even within the same subject domain. "Domain-specific" apparently needs to match writing style, not just subject matter.

## Limitations

- **Small eval set** — 15 hand-written questions is enough to catch a large, consistent gap like the one above, but not a rigorous benchmark. Treat these numbers as directional, not absolute.
- **Same-model judge bias** — the RAGAS judge LLM and the generation LLM are both Groq-hosted models, which can make a judge model somewhat lenient toward outputs in a similar style to its own. This bias is constant across the comparison above, so it doesn't explain the gap between the two embedding models, but the absolute scores should be read with that caveat in mind.
- **Heading-detection heuristic** — chunk boundaries are detected using font-weight metadata (bold + short line = likely a subheading), tuned for this specific book's scanned/OCR'd layout. It will need retuning for a differently-formatted source document.

## Notes

- The Chroma collection must be rebuilt if you change the embedding model.
- Different embedding models produce different vector dimensions, so old vectors and new vectors cannot be mixed in the same collection.