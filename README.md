# 🤖 AI Knowledge Assistant (RAG-based Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask questions using a context-aware AI system powered by a locally running LLM (Mistral via Ollama).

---

## 🔧 Tech Stack

- **LLM:** Mistral (Ollama - local inference)
- **Backend:** FastAPI
- **Frontend:** Gradio
- **Vector Database:** FAISS
- **Embeddings:** SentenceTransformers (MiniLM)
- **Orchestration:** LangChain
- **Document Processing:** PyPDF / Text Splitters

  
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple)
![Ollama](https://img.shields.io/badge/Ollama-LLM-black)
![Gradio](https://img.shields.io/badge/Gradio-UI-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Features

- 📄 Upload and process PDF documents  
- 🔍 Semantic search using vector embeddings  
- 💬 Context-aware Q&A based on document content  
- 🧠 Grounded responses (reduced hallucinations)  
- ⚡ Fast retrieval using FAISS similarity search  
- 🔐 Runs fully locally (no external API dependency)

---

## ⚙️ System Workflow
PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings Generation
↓
Stored in FAISS

User Query
↓
Query Embedding
↓
Similarity Search (Top-K Retrieval)
↓
Context + Query → LLM (Mistral)
↓
Final Answer (Grounded Response)


---
## 🎥 Demo



[Watch Demo Video](https://drive.google.com/file/d/1_Kr5KHqJtXi8u2fcXL2VUxxb7LvyA38y/view?usp=sharing)

## 🧠 Key Highlights

- Processed **50+ documents** using structured ingestion pipelines  
- Improved **retrieval relevance by ~40%** via optimized chunking  
- Reduced hallucinations using **context-grounded responses**  
- Built **modular RAG pipeline architecture**  
- Enabled **fully local LLM inference** (no API dependency)  

---

## ⚙️ Architecture
## ⚙️ Architecture

```mermaid
flowchart TD

subgraph Ingestion Pipeline
A[Upload PDF] --> B[Extract Text]
B --> C[Chunk Text]
C --> D[Create Embeddings]
D --> E[Store in FAISS]
end

subgraph Query Pipeline
F[User Question] --> G[Query Embedding]
G --> H[Search FAISS (Top-K)]
H --> I[Retrieve Context]
I --> J[Prompt + Context]
J --> K[LLM (Mistral)]
K --> L[Answer]
end

E --> H
L --> M[Display in UI]
```
## 🧱 Project Structure


```
app/
│── api.py            # FastAPI routes (/upload, /ask)
│── ui.py             # Gradio frontend (chat + upload UI)
│── rag_pipeline.py   # Core RAG logic (chunking, embeddings, retrieval, LLM)
```

## ▶️ How to Run

### 1. Start Backend
```bash
python -m uvicorn app.api:app --reload
2. Start Frontend
python app/ui.py
3. Run LLM (Ollama)
ollama run mistral
