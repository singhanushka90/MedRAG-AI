from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"


def load_llm():
    model = ChatGroq(
        model=MODEL_NAME,
        temperature=0
    )
    return model


def generate_answer(query, documents):

    llm = load_llm()

    context = "\n\n".join(
        document["content"]
        for document in documents
    )

    prompt = f"""
    You are a medical document question-answering assistant.

    Answer the question using ONLY the provided context.

    If the answer is not present in the context,
    clearly say that the information is not available.

    Do not invent medical information.
    Do not provide diagnosis or treatment advice.
    Treat the documents as synthetic or informational
    unless explicitly stated otherwise.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content