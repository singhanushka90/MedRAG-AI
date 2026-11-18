from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.text_splitter import split_documents
from src.retrieval.vector_store import create_vector_store


def main():

    # Step 1: Load PDF
    documents = load_all_pdfs()

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Store in ChromaDB
    vector_store = create_vector_store(chunks)

    # Step 4: Check stored records
    print("\nTotal stored vectors:")

    print(
        vector_store._collection.count()
    )


if __name__ == "__main__":
    main()