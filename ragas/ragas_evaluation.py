# ── 5. RAGAS evaluation ──────────────────────────────────────────────────

def run_ragas_evaluation(eval_samples: list[dict]):
    """
    eval_samples: list of {"question": str, "reference": str (optional, "" if none)}
    Runs each question through rag_answer(), then scores the results with RAGAS.
    """
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        FactualCorrectness,
    )

    ragas_llm = LangchainLLMWrapper(
        ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    )

    eval_rows = []
    for item in eval_samples:
        question = item["question"]
        reference = item.get("reference", "")
        answer, retrieved_docs, _ = rag_answer(question)
        eval_rows.append(
            {
                "user_input": question,
                "retrieved_contexts": retrieved_docs,
                "response": answer,
                "reference": reference,
            }
        )
        print(f"\nQ: {question}\nA: {textwrap.shorten(answer, 200)}")

    evaluation_dataset = EvaluationDataset.from_list(eval_rows)

    # Faithfulness: checks whether the answer is supported by the retrieved context.
    # Response relevancy: checks whether the answer directly addresses the question.
    # Context precision: checks whether the retrieved chunks are relevant to the question.
    reference_free_metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]

    # Context recall: checks whether the retrieved context contains the information
    # needed to produce the reference answer.
    # Factual correctness: checks whether the generated answer agrees with the reference.
    reference_metrics = [LLMContextRecall(), FactualCorrectness()]

    # Evaluate reference-free metrics for every sample. Reference-based metrics are
    # evaluated only for rows with a trusted reference instead of disabling them for
    # the whole dataset when one unrelated or unanswered question has no reference.
    score_frames = []
    if eval_rows:
        reference_free_dataset = EvaluationDataset.from_list(
            [
                {key: value for key, value in row.items() if key != "reference"}
                for row in eval_rows
            ]
        )
        reference_free_result = evaluate(
            dataset=reference_free_dataset,
            metrics=reference_free_metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        reference_free_df = reference_free_result.to_pandas()
        reference_free_df.insert(0, "evaluation_index", range(len(eval_rows)))
        score_frames.append(reference_free_df)

    reference_indices = [
        index for index, row in enumerate(eval_rows) if row["reference"].strip()
    ]
    if reference_indices:
        reference_dataset = EvaluationDataset.from_list(
            [eval_rows[index] for index in reference_indices]
        )
        reference_result = evaluate(
            dataset=reference_dataset,
            metrics=reference_metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        reference_df = reference_result.to_pandas()
        reference_df.insert(0, "evaluation_index", reference_indices)
        score_frames.append(reference_df)
    else:
        print(
            "\nNote: no samples have a `reference` answer, so context_recall and "
            "factual_correctness cannot be calculated."
        )

    import pandas as pd

    df = pd.DataFrame({"evaluation_index": range(len(eval_rows))}).set_index(
        "evaluation_index"
    )
    for score_frame in score_frames:
        # RAGAS pandas conversion carries the source eval rows back into the frame.
        # Those columns overlap with the per-row df shape we built for the join, so
        # drop them before attaching metric scores to the evaluation ledger.
        score_frame = score_frame.drop(
            columns=["user_input", "retrieved_contexts", "response"],
            errors="ignore",
        )
        score_frame = score_frame.set_index("evaluation_index")
        df = df.join(score_frame, how="left")
    df = df.reset_index()

    print("\n=== RAGAS RESULTS ===")
    print(df.to_string())
    return df

