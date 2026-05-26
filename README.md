st.set_page_confit(page_title="Medical Report AI",layout="wide")
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stTextInput > div > div > input {
    border-radius: 10px;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)
st.title("🩺 Medical Report AI")

An AI-powered Medical Report Analyzer built using **Streamlit + LangChain + Groq + ChromaDB**.

This application allows users to upload medical PDFs and ask questions from the report using **Hybrid RAG (Retrieval-Augmented Generation)**.

---

# 🚀 Features

✅ Upload Multiple PDFs  
✅ AI-Powered Medical Report Explanation  
✅ Hybrid Search (Dense + Sparse Retrieval)  
✅ Chroma Vector Database  
✅ Chat Memory  
✅ Semantic Search  
✅ Simple Medical Explanation  
✅ Session-Based Conversations  
✅ Streamlit Chat UI  
✅ Fast Responses using Groq LLM

---

# 🧠 Tech Stack

| Technology | Purpose |
|---|---|
| Streamlit | Frontend UI |
| LangChain | RAG Pipeline |
| Groq | LLM Inference |
| ChromaDB | Vector Database |
| HuggingFace Embeddings | Text Embeddings |
| BM25 Retriever | Sparse Retrieval |
| PyPDFLoader | PDF Parsing |

---

# 🏗️ System Architecture

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings Generation
    ↓
Chroma Vector Store
    ↓
Hybrid Retrieval
(Dense + Sparse)
    ↓
Groq LLM
    ↓
AI Response
```

---

# 📂 Project Structure

```bash
Medical-Report-AI/
│
├── medical_repo.py
├── requirements.txt
├── README.md

```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Medical-Report-AI.git
```

---

## 2️⃣ Move Into Folder

```bash
cd Medical-Report-AI
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 🔑 Setup API Key

Get your Groq API Key from:

https://console.groq.com/keys

Paste it inside the sidebar.

---

# 📸 Screenshots

Add screenshots inside:

```bash
assets/
```

Example:

```markdown
![App Screenshot](assets/screenshot.png)
```

---

# 📌 Example Questions

- Summarize this report
- Explain abnormal findings
- What precautions should I take?
- Explain blood pressure results
- Is there anything serious?

---

# 🧪 Future Improvements

- OCR Support for Scanned PDFs
- Voice Assistant
- Report Classification
- Download AI Summary PDF
- Multi-LLM Support
- Dark/Light Theme
- Medical Charts Visualization

---

# ⚠️ Disclaimer

This project is for educational purposes only.

Always consult a qualified doctor for medical advice.

---

# 👩‍💻 Author

Anushka Singh

Built with ❤️ using AI & Python
