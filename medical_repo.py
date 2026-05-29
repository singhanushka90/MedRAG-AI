import uuid
import streamlit as st

import tempfile

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


st.set_page_config(page_title="Medical Report AI", layout="wide")
st.title("🩺 Medical Report AI")
st.markdown("Powered by Groq + Hybrid Search + Chroma")

with st.sidebar:
    groq_key = st.text_input("Enter Groq API Key", type="password")
    st.markdown("""
## How It Works
1. Upload PDF
2. Text Extraction
3. Chunking
4. Embedding Creation
5. Hybrid Retrieval
6. AI Answer Generation
""")

pdf_files = st.file_uploader("Upload Medical PDF", type="pdf", accept_multiple_files=True)


if "store" not in st.session_state:
    st.session_state.store = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


if groq_key and pdf_files:
    with st.spinner("Processing reports..."):
        all_docs = []
        for pdf in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(pdf.getvalue())
                temp.flush()
                loader = PyPDFLoader(temp.name)
                all_docs.extend(loader.load())
                st.write("Uploaded File :",pdf.name)

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        if len(chunks) == 0:
            st.error("No readable text found in PDF.")
            st.stop()

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(chunks, embeddings)
        dense = vector_db.as_retriever(search_kwargs={"k":4})
        sparse = BM25Retriever.from_documents(chunks)
        sparse.k = 4

        retriever = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.6, 0.4])
        st.session_state.vector_db = retriever
        st.success("Documents embedded successfully!")


if st.session_state.vector_db and groq_key:

    llm = ChatGroq(api_key=groq_key, model="openai/gpt-oss-120b")

    prompt = ChatPromptTemplate.from_messages([
        ("system","You are an intelligent medical report assistant.\n"
         "Explain reports in very simple English.\n"
         "Do not use complex medical jargon.\n"
         "If the report is unrelated to health, clearly say it.\n"
         "If answer is not found in report, say 'Information not available in report'.\n"
         "Always provide:\n"
         "- Summary\n"
         "- Important Findings\n"
         "- Lifestyle Advice\n"
         "- Doctor Consultation Reminder\n"),
        ("human",
         "Medical Report Context:\n{context}\n\nQuestion: {input}")
    ])

    chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(st.session_state.vector_db, chain)

    chatbot = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    query = st.chat_input("Ask about your report...")
    if query:
        result = chatbot.invoke(
            {"input": query},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(result["answer"])

else:
    st.info("Please upload PDF and enter Groq API key.")







