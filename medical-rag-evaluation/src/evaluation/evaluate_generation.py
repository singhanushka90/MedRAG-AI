from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.retrieval.hybrid_retriever import retrieve_hybrid_documents
from src.reranking.reranker import rerank_documents
from src.generation.llm_generator import generate_answer
from src.evaluation.groq_judge import GroqJudge
from src.evaluation.dataset import 
from deepeval.metric import AnswerRelevancyMetric 


faithfulness_metric = FaithfulnessMetric(
    model=GroqJudge(),
    include_reason=False
)
answer_relevancy=AnswerRelevancyMetric(model=GroqJudge(),include_reason=False)
dataset = load_golden_dataset()

faithfulness_scores = []

for item in dataset:

    query = item["input"]

    # Retrieval
    documents = retrieve_hybrid_documents(query, top_k=3)

    # Reranking
    reranked_documents = rerank_documents(
        query,
        documents,
        top_k=3
    )

    # Generation
    answer = generate_answer(
        query,
        reranked_documents
    )

    # Context
    retrieval_context = [
        document["content"]
        for document in reranked_documents
    ]

    # Test case
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieval_context
    )

    # Faithfulness evaluation
    faithfulness_metric.measure(test_case)

    score = faithfulness_metric.score
    faithfulness_scores.append(score)

    print("\nQuestion:", query)
    print("Answer:", answer)
    print("Faithfulness:", score)


average_score = sum(faithfulness_scores) / len(faithfulness_scores)

print("\nAverage Faithfulness:", average_score)