<div align="center">

# 🩺 MedRAG-AI

### <span style="color:#4CAF50;">AI-Powered Medical Report Analysis using Hybrid RAG</span>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>

<img src="https://img.shields.io/badge/Streamlit-AI_App-red?style=for-the-badge&logo=streamlit"/>

<img src="https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge"/>

<img src="https://img.shields.io/badge/ChromaDB-VectorDB-purple?style=for-the-badge"/>

<img src="https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge"/>

</p>

---

### 🚀 Upload Medical PDFs • Ask Questions • Get AI Explanations

</div>

---

# 📌 Overview

**MedRAG-AI** is an advanced AI-powered Medical Report Assistant built using:

- 🧠 LangChain
- ⚡ Groq LLM
- 📚 ChromaDB
- 🔍 Hybrid Retrieval
- 🎨 Streamlit UI

This application allows users to upload medical reports in PDF format and interact with them using natural language.

The system uses **Hybrid RAG (Retrieval-Augmented Generation)** to provide intelligent and context-aware responses from uploaded reports.

---

# ✨ Features

<div align="center">

| Feature | Description |
|---|---|
| 📄 Multi PDF Upload | Upload multiple medical reports |
| 🧠 AI Report Explanation | Easy-to-understand medical insights |
| 🔍 Hybrid Search | Dense + Sparse Retrieval |
| 💬 Conversational Chat | Ask questions naturally |
| 🧾 Chroma Vector DB | Semantic document storage |
| ⚡ Fast LLM Responses | Powered by Groq |
| 🧠 Chat Memory | Session-based conversations |
| 📊 Semantic Retrieval | Context-aware responses |
| 🎨 Professional UI | Streamlit + Custom CSS |

</div>

---

# 🏗️ System Architecture

```text
              ┌────────────────────┐
              │   PDF Uploads      │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Text Extraction    │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Chunking           │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Embeddings         │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Chroma Vector DB   │
              └─────────┬──────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌───────────────┐              ┌────────────────┐
│ Dense Search  │              │ BM25 Retrieval │
└───────┬───────┘              └───────┬────────┘
        └───────────────┬──────────────┘
                        ▼
              ┌────────────────────┐
              │ Hybrid Retriever   │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Groq LLM           │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ AI Response        │
              └────────────────────┘
```

---

# 🧠 Tech Stack

<div align="center">

| Technology | Usage |
|---|---|
| 🐍 Python | Core Programming |
| 🎨 Streamlit | Frontend UI |
| 🔗 LangChain | RAG Pipeline |
| ⚡ Groq | LLM Inference |
| 🧾 ChromaDB | Vector Storage |
| 🤗 HuggingFace | Embeddings |
| 🔍 BM25 | Sparse Retrieval |
| 📄 PyPDFLoader | PDF Parsing |

</div>

---

# 📂 Project Structure

```bash
MedRAG-AI/
│
├── medical_repo.py
├── requirements.txt
├── README.md

```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/MedRAG-AI.git
```

---

## 2️⃣ Navigate to Folder

```bash
cd MedRAG-AI
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Streamlit App

```bash
streamlit run medical_repo.py
```

---

# 🔑 Setup Groq API Key

Get your API key from:

## 🌐 https://console.groq.com/keys

Paste the API key inside the Streamlit sidebar.

---

# 💡 Example Questions

```text
• Summarize this medical report
• Explain abnormal findings
• What lifestyle changes should I follow?
• Explain blood test results
• Is there anything serious?
• What precautions should I take?
```

---

# 📸 Screenshots

## 🖥️ Main Dashboard

```markdown
![Dashboard](assets/screenshot1.png)
```

---

## 💬 AI Chat Interface

```markdown
![Chat UI](assets/screenshot2.png)
```

---

# 🚀 Future Improvements

- ✅ OCR Support for Scanned PDFs
- ✅ Voice Assistant
- ✅ Download AI Summary PDF
- ✅ Medical Report Classification
- ✅ Multi-LLM Support
- ✅ AI Health Recommendations
- ✅ Medical Charts Visualization
- ✅ Report History Storage

---

# ⚠️ Current Limitations

```diff
- Only text-based PDFs are supported
- Scanned/image PDFs are not supported yet
- OCR pipeline will be added in future versions
```

---

# 🧪 Requirements

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-groq
langchain-huggingface
langchain-text-splitters
langchain-classic
chromadb
sentence-transformers
pypdf
rank-bm25
torch
```

---

# 🛡️ Disclaimer

> This project is developed for educational and research purposes only.

⚠️ Always consult a certified medical professional for actual diagnosis and treatment.

---

# 👩‍💻 Author

<div align="center">

## Anushka Singh

💡 AI & Data Science Enthusiast  
🚀 Building AI-Powered Applications with Python & LangChain

</div>

---

<div align="center">

# ⭐ If you like this project, give it a star on GitHub!

Made with ❤️ using Python + AI

</div>
