🩺 MediChat AI – Medical Report Analysis & RAG Chatbot

MediChat AI is an AI-powered medical report analysis chatbot that helps users understand their laboratory reports through document-based question answering and automated summaries.

The application uses Retrieval-Augmented Generation (RAG) with hybrid retrieval to extract relevant information from uploaded medical PDFs and generate context-aware responses using an LLM.

It combines dense vector search and sparse keyword search to improve document retrieval and helps users explore important values, reference ranges, and findings from their reports.

---

✨ Key Features

- Medical PDF Upload: Upload medical laboratory reports in PDF format.
- Text Extraction: Extract text from uploaded PDF documents.
- Text Chunking: Split extracted text into smaller chunks for efficient retrieval.
- Semantic Embeddings: Convert document chunks into vector embeddings.
- Dense Retrieval: Retrieve semantically relevant document chunks using ChromaDB.
- Sparse Retrieval: Use BM25 for keyword-based document search.
- Hybrid Retrieval: Combine dense and sparse retrieval using EnsembleRetriever.
- AI-Powered Responses: Generate context-aware answers using Groq LLM.
- Medical Report Summary: Generate summaries of important laboratory findings.
- Chat History: Maintain conversation history during the application session.
- Retrieved Context: Display retrieved document chunks for transparency.
- Evaluation Data Collection: Store user queries, generated answers, and retrieved context for future evaluation with DeepEval.

---

🏗️ System Architecture

          User
           |
           v
     Upload Medical PDF
           |
           v
      Text Extraction
           |
           v
       Text Chunking
           |
           v
      Generate Embeddings
           |
           v
        ChromaDB
           |
           |
    +------+------+
    |             |
    v             v
 Dense Search   BM25 Search
    |             |
    +------+------+
           |
           v
    Hybrid Retrieval
    (EnsembleRetriever)
           |
           v
   Retrieved Context
           |
           v
    Groq LLM + Prompt
           |
           v
   Generated Response
           |
           v
     MediChat AI UI
           |
           v
   Evaluation Data
           |
           v
    Future DeepEval

---

🛠️ Tech Stack

Technology| Purpose
Python| Core programming language
Streamlit| User interface
LangChain| RAG pipeline and chain orchestration
Groq| LLM inference
ChatGroq| Integration with Groq LLM
Hugging Face| Text embedding generation
Sentence Transformers| Semantic embeddings
ChromaDB| Vector database
BM25| Keyword-based retrieval
EnsembleRetriever| Hybrid retrieval
PyPDFLoader| PDF text extraction
DeepEval| Planned LLM application evaluation

Models Used

LLM: "openai/gpt-oss-120b"

Embedding Model: "sentence-transformers/all-MiniLM-L6-v2"

---

🔍 Retrieval Strategy

MediChat AI uses a hybrid retrieval approach to retrieve relevant information from medical documents.

1. Dense Retrieval

- Converts document chunks into vector embeddings.
- Stores embeddings in ChromaDB.
- Retrieves information based on semantic similarity.
- Helps find relevant information even when the user's wording differs from the report.

2. Sparse Retrieval

- Uses BM25 for keyword-based retrieval.
- Matches important words and medical terms.
- Helps retrieve exact values, test names, and laboratory terminology.

3. Hybrid Retrieval

Combines both retrieval methods using EnsembleRetriever.

Configuration:

- Dense Retriever: ChromaDB
- Sparse Retriever: BM25
- Dense Weight: 0.6
- Sparse Weight: 0.4
- Top-K: 4 for each retriever

The goal is to combine semantic understanding with keyword matching for better document retrieval.

---

📊 LLM Evaluation – DeepEval

MediChat AI includes an evaluation data collection workflow to assess the retrieval and generation stages of its RAG pipeline.

The application stores:

- User input
- Actual generated output
- Retrieved context
- Source document information

Planned Evaluation Metrics

Metric| What It Evaluates
Contextual Precision| Whether relevant retrieved chunks are ranked higher
Contextual Recall| Whether the retrieved context contains the required information
Faithfulness| Whether the generated answer is supported by retrieved context
Answer Relevancy| Whether the answer addresses the user's question

Current Evaluation Progress

- RAG interaction data collection implemented.
- Generated answers and retrieved contexts captured.
- Evaluation dataset can be exported as JSONL.
- Two test interactions collected during initial testing.
- Actual DeepEval metric execution is the next implementation step.

---

🧪 Initial Testing & Observations

The application was tested using a laboratory report containing CBC, lipid profile, vitamin assays, and other biochemical parameters.

Test 1: General Report Summary

Observation:

- Generated a summary of laboratory findings.
- Identified several values outside the stated reference ranges.
- Displayed retrieved document chunks.
- Included a medical consultation disclaimer.

Test 2: Abnormal Values Identification

Observation:

- Identified low platelet count.
- Identified elevated hematocrit.
- Identified abnormal platelet indices.
- Identified low Vitamin B12.
- Identified insufficient Vitamin D.
- Identified slightly elevated uric acid.

Improvement Areas Identified:

- Some abnormal values, including RBC count and basophils, were missed.
- Some retrieved chunks were less relevant than others.
- Relevant chunks were not always ranked at the top.
- Medical interpretations need stronger grounding in the source report.

These observations will be used to improve retrieval quality and answer generation.

---

🚀 Installation & Setup

1. Clone the Repository

git clone <your-repository-url>
cd MediChat-AI

2. Create a Virtual Environment

python -m venv venv

Activate it:

Windows:

venv\Scripts\activate

Linux / macOS:

source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Configure Groq API Key

Get your API key from:

https://console.groq.com/

Enter the API key in the application's Configuration sidebar.

5. Run the Application

streamlit run app.py

The application will open in your browser.

---

📁 Project Structure

MediChat-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── evaluation/
    └── evaluation_data.jsonl

The structure above represents the main application and evaluation artifacts; update it if your repository contains additional modules.

---

🔮 Future Improvements

- Implement DeepEval metrics for automated RAG evaluation.
- Build a verified golden dataset with expected answers.
- Improve contextual precision and retrieval ranking.
- Reduce irrelevant and duplicate retrieved chunks.
- Improve answer faithfulness and reduce hallucinations.
- Add OCR support for scanned medical PDFs.
- Improve document metadata and source tracking.
- Add more comprehensive evaluation test cases.

---

⚠️ Disclaimer

MediChat AI is an educational AI project designed to help users understand information contained in medical laboratory reports.

It does not provide medical diagnoses or replace professional medical advice. Generated responses may contain errors or omissions. Always consult a qualified healthcare professional for clinical interpretation and treatment decisions.

Medical documents may contain sensitive personal information. Use appropriate privacy safeguards and avoid uploading identifiable medical information to external services without suitable consent and protections.

---

👩‍💻 Author

Anushka Singh

AI & Data Science Student

GitHub: "singhanushka90" (https://github.com/singhanushka90)

---

⭐ If you find this project interesting, consider giving it a star!
