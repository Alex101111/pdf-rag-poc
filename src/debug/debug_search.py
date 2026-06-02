from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

results = vectorstore.similarity_search("word", k=10)
print(f"Top 10 results for 'word':\n")
for i, doc in enumerate(results):
    print(f"[{i+1}] {doc.metadata.get('filename')} p.{doc.metadata.get('page', 0)+1}")
    print(f"     {doc.page_content[:150]}")
    print()