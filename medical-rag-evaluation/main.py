from src.retrieval.hybrid_retriever import retrieve_hybrid_documents
from src.reranking.reranker import rerank_documents


query = "What is the RBC count?"

documents = retrieve_hybrid_documents(query, top_k=3)

reranked_documents = rerank_documents(
    query,
    documents,
    top_k=3
)

for index, document in enumerate(reranked_documents, start=1):

    print(f"\nRank {index}")
    print("Score:", document["reranker_score"])
    print("Content:", document["content"])