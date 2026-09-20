Flow in this project
  ingest.py: extracts text chunks and writes them to JSONL
  load_to_chroma.py: reads those chunks and embeds them into Chroma
  orchestrator.py / retrival.py: query the embedded collection and generate answers


  (locally, all-MiniLM-L6-v2) to create the embeddings automatically.

  Confirmed: NeuML/pubmedbert-base-embeddings — real, well-documented, 768-dimensional,