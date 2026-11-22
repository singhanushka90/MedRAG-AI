from src.retrieval.dense_retriever import retrieve_documents
from src.retrieval.sparse_retriever import create_sparse_retriever
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.text_splitter import split_documents


RRF_K = 60


def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    k=RRF_K
):
    scores = {} #RRF SCORE
    documents = {} #ACTUAL CONTENT AND METADATA

    result_lists = [
        dense_results,
        sparse_results
    ]

    for result_list in result_lists:

        for rank, result in enumerate(result_list, start=1):

            if isinstance(result, dict):
                content = result["content"]
                metadata = result["metadata"]
            else:
                content = result.page_content
                metadata = result.metadata

            key = (
                content,
                metadata.get("source"),
                metadata.get("page")
            )

            if key not in scores:
                scores[key] = 0
                documents[key] = {
                    "content": content,
                    "metadata": metadata
                }

            scores[key] += 1 / (k + rank)

    ranked_results = []

    for key, score in scores.items():

        ranked_results.append({
            **documents[key],
            "rrf_score": score
        })

    ranked_results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return ranked_results


def retrieve_hybrid_documents(query, top_k=3):

    dense_results = retrieve_documents(
        query,
        k=top_k
    )

    documents = load_all_pdfs()
    chunks = split_documents(documents)

    sparse_retriever = create_sparse_retriever(chunks)

    sparse_results = sparse_retriever.invoke(query)

    final_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results
    )

    return final_results[:top_k]