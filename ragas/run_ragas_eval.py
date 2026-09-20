"""
Scores ./eval_output/ragas_dataset.jsonl (produced by build_ragas_dataset.py)
using RAGAS metrics, with Groq as the judge LLM.

Metrics used and what each one actually tells you:
  - faithfulness:        Does the generated answer's claims actually appear
                          in the retrieved context, or is the model making
                          things up (hallucinating)? Judges GENERATION.
  - answer_relevancy:     Does the answer actually address the question asked
                          (not off-topic/evasive)? Judges GENERATION.
  - context_precision:    Of the chunks retrieved, how many were actually
                          relevant/useful vs noise? Judges RETRIEVAL.
  - context_recall:       Does the retrieved context contain everything
                          needed to produce the reference answer, or is
                          retrieval MISSING something? Judges RETRIEVAL.
                          (This is the one that will expose whether the
                          cross-referential question in eval_dataset.py
                          needs multi-hop -- see earlier chat.)

Wiring note: RAGAS's `provider="groq"` adapter is broken in the currently
installed ragas version (confirmed bug: it tries to patch Groq's client with
Anthropic-shaped internals). Workaround used here: Groq exposes an
OpenAI-compatible endpoint, so we point AsyncOpenAI's base_url at Groq
instead -- ragas's OpenAI adapter path works correctly with it.

Requires (on top of requirements.txt):
    pip install ragas "langchain-community<0.4.2" openai

Usage:
    python run_ragas_eval.py --dataset ./eval_output/ragas_dataset.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ragas.cache import DiskCacheBackend
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

load_dotenv()

# Using the SAME model as retrival.py's generation calls, per explicit
# request -- the separate-quota-pool idea (llama-3.1-8b-instant) turned out
# to be a dead end: that model is now fully deprecated/removed from Groq
# (confirmed via the 404 model_not_found error), not just "on its way out"
# as the docs suggested. Importing it directly from retrival.py rather than
# hardcoding it a second time, so the two files can't drift out of sync if
# you change the model in retrival.py later.
import retrival
JUDGE_MODEL = retrival.GROQ_MODEL
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# NOTE: since this is now the SAME model as generation, judge calls and
# generation calls share the same daily quota. If you hit the TPD limit
# again, it means you're out of quota for this model entirely today --
# no amount of retrying here will fix that (see the earlier TPD explanation).
# The disk cache below still helps: it prevents wasting more of that shared
# quota on questions you've already scored successfully.

# openai/gpt-oss-120b on Groq's free/on-demand tier has an 8000 TPM cap.
# Each metric call sends the full question + retrieved context + JSON schema
# instructions, so a run of 15 questions x 4 metrics burns through this fast.
# DELAY_BETWEEN_CALLS spaces requests out; MAX_RETRIES + backoff handles the
# rare 429 that still slips through despite the delay.
DELAY_BETWEEN_CALLS_SECONDS = 3.0
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5.0

# Default judge max_tokens is 1024 (confirmed via InstructorModelArgs()) --
# too small for Faithfulness's JSON output when the generated answer has
# many claims to verify, causing "max completion tokens reached before
# generating a valid document" (a real truncation, not a real validation
# failure). Raised here.
JUDGE_MAX_TOKENS = 4096


class ChromaOnnxEmbeddings(BaseRagasEmbedding):
    """Reuses ChromaDB's built-in local embedding model (all-MiniLM-L6-v2 via
    onnxruntime) instead of pulling in sentence-transformers/torch just for
    eval scoring. You already have this exact model cached on disk from
    running load_to_chroma.py -- no new multi-GB dependency needed.

    Only used by RAGAS's answer_relevancy metric (the only one of the four
    that needs embeddings, not the LLM judge).
    """

    def __init__(self):
        super().__init__()
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._ef = DefaultEmbeddingFunction()

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self._ef([text])[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        # DefaultEmbeddingFunction is sync-only (onnxruntime), so just call
        # it directly -- fine for eval-time usage, no need for real async.
        return self.embed_text(text)


def build_judge_llm():
    client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    # DiskCacheBackend: if a question+metric combination was already scored
    # successfully in a previous run, re-running this script reuses that
    # cached result instead of spending quota calling the API again. This
    # matters a lot when you're re-running after hitting a quota wall --
    # without this, EVERY re-run starts back at question 1, re-spending
    # tokens on results you already have.
    cache_dir = PROJECT_ROOT / "dataset" / "eval_output" / ".ragas_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = DiskCacheBackend(cache_dir=str(cache_dir))
    return llm_factory(model=JUDGE_MODEL, provider="openai", client=client,
                        max_tokens=JUDGE_MAX_TOKENS, cache=cache)


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _extract_retry_after_seconds(error_message: str) -> float | None:
    """Groq's 429 body includes 'Please try again in 352.5ms.' -- parse it
    out so we can wait the actual suggested time instead of a blind guess."""
    match = re.search(r"try again in ([\d.]+)(ms|s)", error_message)
    if not match:
        return None
    value, unit = match.groups()
    value = float(value)
    return value / 1000 if unit == "ms" else value


async def score_with_retry(metric, kwargs: dict) -> float | str:
    """Calls metric.ascore() with real retry+backoff. Instructor's own
    internal retry gave up after a single attempt in practice (confirmed:
    'Total attempts: 1' in the error logs), so this wrapper is what actually
    provides resilience against transient 429s and occasional truncated-JSON
    400s from the judge model."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await metric.ascore(**kwargs)
            return round(result.value, 3)
        except Exception as e:
            last_error = e
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str
            retry_after = _extract_retry_after_seconds(error_str) if is_rate_limit else None
            wait = retry_after if retry_after else BASE_BACKOFF_SECONDS * attempt
            if attempt < MAX_RETRIES:
                print(f"    retrying {metric.name} (attempt {attempt}/{MAX_RETRIES}, "
                      f"waiting {wait:.1f}s)...")
                await asyncio.sleep(wait)
    return f"ERROR: {last_error}"


async def score(dataset_path: str):
    judge_llm = build_judge_llm()
    embeddings = ChromaOnnxEmbeddings()

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        ContextPrecisionWithoutReference(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    records = load_dataset(dataset_path)
    print(f"Loaded {len(records)} records from {dataset_path}\n")

    faithfulness_m, answer_relevancy_m, context_precision_m, context_recall_m = metrics

    results = []
    for i, r in enumerate(records, 1):
        row_scores = {"question": r["user_input"]}

        # Each metric takes a DIFFERENT subset of fields -- calling all four
        # with the same kwargs blindly throws a TypeError (verified against
        # the actual installed ragas version before writing this).
        checks = [
            (faithfulness_m, dict(user_input=r["user_input"], response=r["response"],
                                   retrieved_contexts=r["retrieved_contexts"])),
            (answer_relevancy_m, dict(user_input=r["user_input"], response=r["response"])),
            (context_precision_m, dict(user_input=r["user_input"], response=r["response"],
                                        retrieved_contexts=r["retrieved_contexts"])),
            (context_recall_m, dict(user_input=r["user_input"], retrieved_contexts=r["retrieved_contexts"],
                                     reference=r["reference"])),
        ]

        for metric, kwargs in checks:
            row_scores[metric.name] = await score_with_retry(metric, kwargs)
            await asyncio.sleep(DELAY_BETWEEN_CALLS_SECONDS)  # stay under the 8000 TPM cap

        print(f"[{i}/{len(records)}] {r['user_input'][:60]}... -> {row_scores}")
        results.append(row_scores)

    # Aggregate: average each metric across all questions (skipping errors)
    print("\n=== Averages across all questions ===")
    for metric in metrics:
        vals = [r[metric.name] for r in results if isinstance(r[metric.name], (int, float))]
        if vals:
            print(f"  {metric.name}: {sum(vals) / len(vals):.3f}  (n={len(vals)})")
        else:
            print(f"  {metric.name}: no successful scores")

    out_path = dataset_path.replace(".jsonl", "_scored.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPer-question scores written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./eval_output/ragas_dataset.jsonl")
    args = parser.parse_args()
    asyncio.run(score(args.dataset))