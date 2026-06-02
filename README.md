# PDF RAG PoC

A local, privacy-first question-answering app for PDF documents.  
Ask questions in plain language — get answers grounded in your documents, no internet required.

---

## How it works

This app uses a technique called **Retrieval-Augmented Generation (RAG)**:

1. **Ingest** — your PDF files are loaded, split into semantic chunks, and converted into numerical vectors (embeddings) using a local embedding model
2. **Store** — those vectors are saved in a local vector database (ChromaDB) alongside the original text
3. **Query** — when you ask a question, it is converted into a vector and compared against all stored vectors to find the most relevant chunks
4. **Answer** — the relevant chunks are passed to a local language model (phi4-mini) which reads them and writes a grounded answer

Everything runs locally on your machine. Your documents never leave your computer.

```
PDF files
    ↓  PyPDFLoader
Pages with metadata
    ↓  SemanticChunker + nomic-embed-text
Semantic chunks as vectors
    ↓  ChromaDB
Persistent vector index on disk
    ↓  (at query time)
Your question → nomic-embed-text → top 4 matching chunks → phi4-mini → answer
```

---

## Stack

| Component | Tool | Role |
|---|---|---|
| PDF loading | PyPDFLoader (LangChain) | Reads PDFs page by page |
| Chunking | SemanticChunker | Splits text at meaning boundaries |
| Embeddings | nomic-embed-text (Ollama) | Converts text to vectors |
| Vector DB | ChromaDB | Stores and searches vectors |
| LLM | phi4-mini (Ollama) | Reads context and writes answers |
| UI | Streamlit | Browser-based chat interface |

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

---

## Setup

**1 — Clone the repo**
```bash
git clone https://github.com/Alex101111/pdf-rag-poc.git
cd pdf-rag-poc
```

**2 — Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
. .venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

**3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**4 — Pull the required Ollama models**
```bash
ollama pull phi4-mini
ollama pull nomic-embed-text
```

---

## Usage

**Step 1 — Add your PDFs**

Drop your PDF files into the `data/` folder.

**Step 2 — Index your documents**
```bash
python src/ingest.py
```

This reads all PDFs in `data/`, chunks them semantically, embeds them, and stores the index in `vectorstore/`. Run this once per set of documents, and again whenever you add new files.

**Step 3 — Launch the app**
```bash
streamlit run src/app.py
```

Your browser opens at `http://localhost:8501`. Type a question and the app will answer from your documents.

---

## Project structure

```
pdf-rag-poc/
├── data/              # Drop your PDFs here (gitignored)
├── vectorstore/       # ChromaDB index, auto-generated (gitignored)
├── src/
│   ├── ingest.py      # Indexes PDFs into the vector store
│   └── app.py         # Streamlit chat interface
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Notes

- Re-run `ingest.py` whenever you add or update documents — it clears and rebuilds the index
- The first response after launching may be slow while phi4-mini loads into memory
- On CPU-only machines expect 5–15 seconds per answer
- Answers include source citations showing which document and page each chunk came from
