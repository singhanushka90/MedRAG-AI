from langchain_classic.retrievers import BM25Retriever

def create_sparse_retriever(chunks,k=3):

    retriever = BM25Retriever.from_documents(
        chunks
    )

    retriever.k = 3

    return retriever
