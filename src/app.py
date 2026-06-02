import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="PDF RAG", page_icon="📄")
st.title("📄 PDF Assistant")
st.caption("Ask questions about your documents")

@st.cache_resource
def load_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = OllamaLLM(model="phi4-mini", temperature=0)

    prompt = PromptTemplate.from_template("""
You are an assistant that answers questions based on the provided documents.
Use ONLY the context below. If the answer is not there, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('filename', 'unknown')} p.{doc.metadata.get('page', '?')+1}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# Load chain
try:
    chain, retriever = load_chain()
except Exception as e:
    st.error(f"Could not load vectorstore: {e}")
    st.info("Run `python src/ingest.py` first to index your PDFs.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            sources = retriever.invoke(question)

        st.write(answer)

        # Show sources
        with st.expander("📎 Sources"):
            for doc in sources:
                st.caption(
                    f"📄 {doc.metadata.get('filename', 'unknown')} "
                    f"— page {doc.metadata.get('page', '?')+1}"
                )
                st.text(doc.page_content[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})