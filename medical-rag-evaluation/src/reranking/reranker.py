from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"


def load_reranker():
    model = CrossEncoder(MODEL_NAME)
    return model


def rerank_documents(query, documents, top_k=3):

    model = load_reranker()

    pairs = [
        (query, document["content"])
        for document in documents
    ]

    scores = model.predict(pairs)

    reranked_documents = []

    for document, score in zip(documents, scores):

        reranked_documents.append({
            **document,
            "reranker_score": float(score)
        })

    reranked_documents.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_documents[:top_k]