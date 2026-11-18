from src.retrieval.vector_store import load_vector_store


def retrieve_documents(query, k=3):

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    retrieved_documents = []

    for document, score in results:

        retrieved_documents.append({
            "content": document.page_content,
            "metadata": document.metadata,
            "score": score
        })

    return retrieved_documents