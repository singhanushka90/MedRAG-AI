from src.retrieval.hybrid_retriever import retrieve_hybrid_documents


def main():

    query = "What is the haemoglobin level?"

    results = retrieve_hybrid_documents(query, top_k=3)

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("Content:", result["content"])
        print("RRF Score:", result["rrf_score"])
        print("Metadata:", result["metadata"])


if __name__ == "__main__":
    main()