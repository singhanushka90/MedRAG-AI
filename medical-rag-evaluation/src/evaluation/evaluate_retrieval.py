from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric
)

from deepeval.test_case import LLMTestCase

from src.evaluation.dataset import load_golden_dataset
from src.evaluation.groq_judge import GroqJudge
from src.retrieval.hybrid_retriever import retrieve_hybrid_documents
from src.reranking.reranker import rerank_documents


def evaluate_retrieval():

    dataset = load_golden_dataset()

    judge = GroqJudge()

    recall_metric = ContextualRecallMetric(
        model=judge,
        threshold=None,
        include_reason=False,
        async_mode=False
    )

    precision_metric = ContextualPrecisionMetric(
        model=judge,
        threshold=None,
        include_reason=False,
        async_mode=False
    )

    recall_scores = []
    precision_scores = []

    for index, item in enumerate(dataset, start=1):

        query = item["input"]

        print(f"\n{'=' * 50}")
        print(f"Test Case {index}: {query}")

        
        retrieved_documents = retrieve_hybrid_documents(
            query,
            top_k=3
        )
        retrieved_documents=rerank_documents(query,retrieved_documents,top_k=3)

        retrieval_context = [
            document["content"]
            for document in retrieved_documents
        ]

       
        test_case = LLMTestCase(
            input=query,

            # Placeholder: this project currently evaluates
            # retrieval only, not answer generation.
            actual_output=item["expected_output"],

            expected_output=item["expected_output"],

            context=item["expected_context"],

            retrieval_context=retrieval_context
        )

        
        recall_metric.measure(test_case)

        recall_score = recall_metric.score
        recall_scores.append(recall_score)

        print("\nContextual Recall:", recall_score)
        print("Recall Reason:", recall_metric.reason)

       
        precision_metric.measure(test_case)

        precision_score = precision_metric.score
        precision_scores.append(precision_score)

        print("\nContextual Precision:", precision_score)
        print("Precision Reason:", precision_metric.reason)

    print("\n" + "=" * 50)
    print("FINAL RETRIEVAL EVALUATION")

    print(
        "Average Contextual Recall:",
        sum(recall_scores) / len(recall_scores)
    )

    print(
        "Average Contextual Precision:",
        sum(precision_scores) / len(precision_scores)
    )


if __name__ == "__main__":
    evaluate_retrieval()