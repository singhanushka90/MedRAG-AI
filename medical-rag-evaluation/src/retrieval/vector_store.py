from langchain_chroma import Chroma

from src.embeddings.embedding_model import load_embedding_model

CHROMA_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "medical_documents"


def create_vector_store(chunks):

    embeddings = load_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIRECTORY
    )

    print("Medical documents stored in ChromaDB.")

    return vector_store

def load_vector_store():

    embeddings = load_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIRECTORY
    )

    return vector_store