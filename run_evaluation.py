# eval/run_evaluation.py
# Run RAGAS evaluation on the golden test set.
#
# Usage:
#   python eval/run_evaluation.py
#
# What this does:
# 1. Loads the golden test set (questions + expected answers)
# 2. Runs each question through the full RAG pipeline
# 3. Collects: question, generated answer, retrieved contexts, ground truth
# 4. Runs RAGAS metrics on all results
# 5. Saves scores to eval/results_{date}.json
# 6. Prints a summary with pass/fail per metric

import os
import sys
import json
from datetime import datetime
from pathlib import Path

from sympy import evaluate

# Add parent directory to path so we can import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# All imports at module level — never inside functions
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from retriever import retrieve_and_rerank
from dotenv import load_dotenv

load_dotenv()

# ── Thresholds — what scores are "good enough" ───────────────────────
# Adjust these based on your quality requirements
# These are reasonable starting points for a Wikipedia-based RAG system
THRESHOLDS = {
    "faithfulness": 0.75,       # 75% of claims grounded in context
    "answer_relevancy": 0.75,   # 75% of answer addresses the question
    "context_precision": 0.60,  # 60% of retrieved chunks are useful
    "context_recall": 0.60,     # 60% of needed facts were retrieved
}


def build_rag_chain(vectorstore, llm):
    """Build the RAG chain for evaluation."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
Use ONLY the following retrieved documents to answer the question.
If the answer is not in the documents, say you don't know.

Retrieved documents:
{context}"""),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()


def run_evaluation():
    print("=" * 60)
    print("RAG EVALUATION — RAGAS Framework")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load golden test set ─────────────────────────────────────────
    golden_path = Path(__file__).parent / "golden_set.json"
    with open(golden_path) as f:
        test_cases = json.load(f)
    test_cases = test_cases[:2]
    print(f"\nLoaded {len(test_cases)} test cases from golden set")

    # ── Load vector store ────────────────────────────────────────────
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    doc_count = vectorstore._collection.count()
    print(f"Vector store loaded. Documents indexed: {doc_count}")

    # ── Load LLM ─────────────────────────────────────────────────────
    groq_key = os.getenv("groq_api_key")
    if not groq_key:
        # Try loading from .env file
        from dotenv import load_dotenv
        load_dotenv()
        groq_key = os.getenv("groq_api_key")

    if not groq_key:
        print("ERROR: groq_api_key not found. Set it in .env file.")
        return

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_key,
        temperature=0  # Deterministic for evaluation
    )
    chain = build_rag_chain(vectorstore, llm)

    # ── Run each test case through the pipeline ───────────────────────
    print(f"\nRunning {len(test_cases)} questions through RAG pipeline...")
    print("-" * 60)

    results = []
    for i, case in enumerate(test_cases):
        question = case["question"]
        print(f"[{i+1}/{len(test_cases)}] {question[:60]}...")

        try:
            # Retrieve with reranker
            docs = retrieve_and_rerank(
                query=question,
                vectorstore=vectorstore,
                initial_k=20,
                final_k=4
            )

            if not docs:
                print(f"  WARNING: No documents retrieved for this question")
                contexts = ["No relevant documents found"]
            else:
                contexts = [doc.page_content for doc in docs]

            # Format context for LLM
            context = "\n\n".join([
                f"[Doc {j+1}]: {doc.page_content}"
                for j, doc in enumerate(docs)
            ])

            # Generate answer
            answer = chain.invoke({
                "context": context,
                "question": question
            })

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": case["expected_answer"]
            })

            print(f"  Answer: {answer[:80]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            # Add failed case with empty answer so RAGAS can still run
            results.append({
                "question": question,
                "answer": "Error generating answer",
                "contexts": ["Error during retrieval"],
                "ground_truth": case["expected_answer"]
            })

    # ── Run RAGAS evaluation ──────────────────────────────────────────
    print(f"\n{'-' * 60}")
    print("Running RAGAS evaluation...")

    # Tell RAGAS to use Groq instead of OpenAI for evaluation
    # RAGAS needs an LLM to judge answer quality and an embedding model
    # to measure semantic similarity
    eval_llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("groq_api_key"),
        temperature=0
    ))

    eval_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ))

    # Configure each metric to use our LLM and embeddings
    faithfulness.llm = eval_llm
    answer_relevancy.llm = eval_llm
    answer_relevancy.embeddings = eval_embeddings
    context_precision.llm = eval_llm
    context_recall.llm = eval_llm

    dataset = Dataset.from_list(results)

    scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

    # ── Print results ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")

    scores_dict = dict(scores)
    all_passed = True

    for metric, score in scores_dict.items():
        threshold = THRESHOLDS.get(metric, 0.70)
        passed = score >= threshold
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        print(f"\n{metric}:")
        print(f"  Score:     {score:.4f}  [{bar}]")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Status:    {status}")

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL METRICS PASSED ✓' if all_passed else 'SOME METRICS FAILED ✗'}")
    print(f"{'=' * 60}")

    # ── Save results ──────────────────────────────────────────────────
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"eval_{timestamp}.json"

    eval_output = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(test_cases),
        "scores": scores_dict,
        "thresholds": THRESHOLDS,
        "all_passed": all_passed,
        "per_question_results": results
    }

    with open(results_file, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return scores_dict


if __name__ == "__main__":
    run_evaluation()