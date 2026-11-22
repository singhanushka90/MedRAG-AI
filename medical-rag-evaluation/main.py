from src.retrieval.hybrid_retriever import retrieve_hybrid_documents
from src.reranking.reranker import rerank_documents
from src.generation.llm_generator import generate_answer


query = "What is the haemoglobin level in the report?"

documents = retrieve_hybrid_documents(query, top_k=3)

reranked_documents = rerank_documents(
    query,
    documents,
    top_k=3
)

answer = generate_answer(
    query,
    reranked_documents
)

print("\nFinal Answer:")
print(answer)