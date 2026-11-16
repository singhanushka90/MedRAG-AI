import uuid
import os
import json
import tempfile

import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers import (EnsembleRetriever,BM25Retriever)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables.history import (RunnableWithMessageHistory)

from langchain_core.chat_history import InMemoryChatMessageHistory

st.set_page_config(
    page_title="MediChat AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MediChat AI")
st.markdown("Powered by Groq + Hybrid Retrieval + Chroma")


defaults = {
    "store": {},
    "session_id": str(uuid.uuid4()),
    "retriever": None,
    "vector_store": None,
    "documents_processed": False,
    "evaluation_data": [],
    "chatbot": None,
    "groq_key": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_session_history(session_id):

    if session_id not in st.session_state.store:

        st.session_state.store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return st.session_state.store[session_id]


with st.sidebar:

    st.header("Configuration")

    groq_key = st.text_input(
        "Enter Groq API Key",
        type="password"
    )

    st.markdown("---")

    st.markdown("""
    ### How It Works

    1. Upload Medical PDF
    2. Text Extraction
    3. Chunking
    4. Embeddings
    5. Hybrid Retrieval
    6. LLM Generation
    7. DeepEval Evaluation
    """)

    st.markdown("---")

    if st.button("Clear Chat History"):

        st.session_state.store = {}
        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()


pdf_files = st.file_uploader(
    "Upload Medical PDF",
    type=["pdf"],
    accept_multiple_files=True
)


if st.button("Process Documents", type="primary"):

    if not pdf_files:

        st.error("Please upload at least one PDF.")

    else:

        try:

            with st.spinner("Processing medical reports..."):

                all_docs = []

                for pdf in pdf_files:

                    temp_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as temp:

                            temp.write(pdf.getvalue())
                            temp_path = temp.name

                        loader = PyPDFLoader(temp_path)

                        docs = loader.load()

                        for doc in docs:

                            doc.metadata["source"] = pdf.name

                            doc.metadata["page_number"] = (
                                doc.metadata.get("page", 0) + 1
                            )

                        all_docs.extend(docs)

                    finally:

                        if temp_path and os.path.exists(temp_path):

                            os.unlink(temp_path)


                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(all_docs)

                if not chunks:

                    st.error("No readable text found in PDF.")
                    st.stop()


                # EMBEDDINGS

                embeddings = HuggingFaceEmbeddings(
                    model_name=(
                        "sentence-transformers/all-MiniLM-L6-v2"
                    )
                )

                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                dense_retriever = vector_store.as_retriever(
                    search_kwargs={"k": 4}
                )

                sparse_retriever = BM25Retriever.from_documents(
                    chunks
                )

                sparse_retriever.k = 4

                hybrid_retriever = EnsembleRetriever(
                    retrievers=[
                        dense_retriever,
                        sparse_retriever
                    ],
                    weights=[0.6, 0.4]
                )


                st.session_state.vector_store = vector_store

                st.session_state.retriever = hybrid_retriever

                st.session_state.documents_processed = True

                st.session_state.chatbot = None

                st.session_state.evaluation_data = []

                st.success(
                    f"Successfully processed {len(pdf_files)} PDFs "
                    f"and created {len(chunks)} chunks!"
                )

        except Exception as e:

            st.error(f"Document processing failed: {str(e)}")

if (
    st.session_state.documents_processed
    and st.session_state.retriever is not None
):

    if groq_key:

        if st.session_state.groq_key != groq_key:

            st.session_state.groq_key = groq_key
            st.session_state.chatbot = None


        if st.session_state.chatbot is None:

            llm = ChatGroq(
                api_key=groq_key,
                model="openai/gpt-oss-120b",
                temperature=0
            )

            prompt = ChatPromptTemplate.from_messages([

                (
                    "system",
                    """
                    You are MediChat, a medical report explanation assistant.

                    Your responsibilities:

                    1. Explain medical reports in simple English.
                    2. Use only information supported by the provided context.
                    3. Never invent medical values or findings.
                    4. If information is missing, say:
                       "Information not available in the report."
                    5. Do not diagnose diseases.
                    6. Do not prescribe medicines or treatments.
                    7. Do not make unsupported lifestyle recommendations.
                    8. Encourage users to consult qualified doctors
                       for medical interpretation.

                    If the question is unrelated to the uploaded report,
                    clearly explain that.

                    Provide concise answers and mention relevant
                    report findings where available.

                    Medical Context:

                    {context}
                    """
                ),

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),

                (
                    "human",
                    "{input}"
                )

            ])

            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )


            rag_chain = create_retrieval_chain(
                st.session_state.retriever,
                document_chain
            )

            chatbot = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            st.session_state.chatbot = chatbot

        history = get_session_history(
            st.session_state.session_id
        )

        for message in history.messages:

            if message.type == "human":

                with st.chat_message("user"):
                    st.write(message.content)

            elif message.type == "ai":

                with st.chat_message("assistant"):
                    st.write(message.content)

        query = st.chat_input(
            "Ask about your medical report..."
        )

        if query:

            with st.chat_message("user"):
                st.write(query)

            with st.spinner("Analyzing your report..."):

                try:

                    result = st.session_state.chatbot.invoke(

                        {"input": query},

                        config={
                            "configurable": {
                                "session_id": (
                                    st.session_state.session_id
                                )
                            }
                        }
                    )


                    answer = result["answer"]

                    retrieved_docs = result["context"]


                    retrieval_context = [

                        doc.page_content

                        for doc in retrieved_docs

                    ]

                    sources = [

                        {
                            "source": doc.metadata.get(
                                "source", "Unknown"
                            ),

                            "page": doc.metadata.get(
                                "page_number", "Unknown"
                            )

                        }

                        for doc in retrieved_docs

                    ]

                    evaluation_record = {

                        "input": query,

                        "actual_output": answer,

                        "retrieval_context": retrieval_context,

                        "sources": sources

                    }

                    st.session_state.evaluation_data.append(
                        evaluation_record
                    )


                    with st.chat_message("assistant"):

                        st.write(answer)

                        with st.expander("View Retrieved Context"):

                            for i, doc in enumerate(
                                retrieved_docs
                            ):

                                st.markdown(
                                    f"### Retrieved Chunk {i+1}"
                                )

                                st.write(doc.page_content)

                                st.caption(
                                    f"Source: {doc.metadata.get('source')} "
                                    f"| Page: {doc.metadata.get('page_number')}"
                                )


                except Exception as e:

                    st.error(f"Chat error: {str(e)}")


    else:

        st.info("Please enter your Groq API key.")


else:

    st.info(
        "Upload medical PDFs and click Process Documents "
        "to start MediChat."
    )


if st.session_state.evaluation_data:

    st.markdown("---")

    st.subheader("📊 Evaluation Data")

    st.write(
        f"Total test interactions: "
        f"{len(st.session_state.evaluation_data)}"
    )

    jsonl_data = "\n".join(

        json.dumps(record, ensure_ascii=False)

        for record in st.session_state.evaluation_data

    )

    st.download_button(

        label="Download Evaluation Dataset (JSONL)",

        data=jsonl_data,

        file_name="medichat_evaluation.jsonl",

        mime="application/json"

    )