import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import shutil

def ingest(pdf_folder="data", vectorstore_dir="vectorstore"):
    # Find PDFs
    pdf_files = [
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDFs found in data/ folder.")
        return

    print(f"Found {len(pdf_files)} PDF(s)\n")

    # Load pages
    all_pages = []
    for path in pdf_files:
        loader = PyPDFLoader(path)
        pages = loader.load()
        filename = os.path.basename(path)
        for page in pages:
            page.metadata["filename"] = filename
            page.metadata["file_type"] = "pdf"
            page.metadata["total_pages"] = len(pages)
        all_pages.extend(pages)
        print(f"  {filename} → {len(pages)} pages")

    print(f"\nTotal pages: {len(all_pages)}")

    # Semantic chunking
    print("\nChunking with semantic splitter...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    chunks = splitter.split_documents(all_pages)
    print(f"Created {len(chunks)} chunks")

    # Clear old vectorstore
    if os.path.exists(vectorstore_dir):
        shutil.rmtree(vectorstore_dir)

    # Store
    print("\nEmbedding and storing...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vectorstore_dir
    )
    print(f"\nDone! {len(chunks)} chunks stored.")

if __name__ == "__main__":
    ingest()