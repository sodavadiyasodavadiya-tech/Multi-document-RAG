---
title: DocuMind - Intelligent RAG Chatbot
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🧠 DocuMind: Intelligent RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-green)
![Groq](https://img.shields.io/badge/LLM-Llama_3.1-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-yellow)

**DocuMind** is a state-of-the-art Retrieval-Augmented Generation (RAG) system designed to interact with your documents intelligently. It combines the power of **Google's Gemini 2.5 Flash** for high-speed, accurate reasoning with a **Hybrid Search** mechanism (Dense Vectors + BM25) for precise retrieval.

## ✨ Key Features

- **🚀 Dual-Engine Intelligence**:
  - Primary: **Gemini 2.5 Flash** (Fast & Accurate)
  - Fallback: **Groq Llama 3.1** (Ultra-low latency backup)
- **🔍 Hybrid Search Architecture**: Combines **FAISS** (Dense vector search) with **BM25** (Keyword matching) for superior retrieval accuracy.
- **📄 Multi-Format Support**:
  - PDFs, DOCX, PPTX
  - Text Files (TXT, MD)
  - Images (OCR supported)
- **⚡ Modern UI**: sleek **Glassmorphism** design built with Streamlit, featuring glowing animations and responsive layouts.
- **🛠️ Robust Backend**:
  - **FastAPI** endpoints for programmatic access.
  - **Streamlit** for interactive user experience.
  - **BGE-Small** embeddings for efficient semantic representation.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Custom CSS & Glassmorphism UI)
- **LLM Orchestration**: Python (Custom Router with Fallback)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace BGE-Small
- **Document Processing**: Unstructured, PyMuPDF, Python-docx

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/documind.git
cd documind
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
# Ensure faiss-cpu is installed
pip install faiss-cpu
```

### 3. Configure Environment Keys
Create a `.env` file in the root directory and add your API keys:
```env
GEMINI_API_KEY=your_google_gemini_key
GROQ_API_KEY=your_groq_api_key
```

## 🏃‍♂️ Usage

### Option A: Run the Web Interface (Streamlit)
Launch the interactive chatbot:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Run the API Server (FastAPI)
Start the backend API for programmatic document processing:
```bash
uvicorn app:app --reload
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

## 📂 Project Structure

```
├── data/                   # Storage for uploads & vector index
├── app.py                  # FastAPI Backend
├── streamlit_app.py        # Streamlit Frontend (Main UI)
├── rag_pipeline.py         # Core RAG Logic (Chunking, Embedding, Retrieval)
├── vectordb.py             # FAISS + BM25 Vector Store Implementation
├── llm_router.py           # LLM Routing Logic (Gemini <-> Groq)
├── requirements.txt        # Python Dependencies
└── README.md               # Project Documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to verify the `dev` branch and pull requests.

## 📄 License

This project is licensed under the MIT License.
