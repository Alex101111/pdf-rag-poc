import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="PDF RAG", page_icon="📄")
st.title("📄 PDF Assistant")
st.caption("Ask questions about your documents")

@st.cache_resource
def load_components():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = OllamaLLM(model="phi4-mini", temperature=0)
    return retriever, llm

try:
    retriever, llm = load_components()
except Exception as e:
    st.error(f"Could not load vectorstore: {e}")
    st.info("Run `python src/ingest.py` first to index your PDFs.")
    st.stop()

# Prompt now includes conversation history
prompt = PromptTemplate.from_template("""
You are an assistant for PDF document analysis.
Use ONLY the context below to answer. Do not use prior knowledge.
Always mention the document name and page number when referencing information.
If multiple documents contain relevant information, summarise all of them.
If the answer is not in the context, say exactly: "I could not find that information in the provided documents."

Previous conversation:
{history}

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('filename', 'unknown')} p.{doc.metadata.get('page', '?')+1}]\n{doc.page_content}"
        for doc in docs
    )

def format_history(messages):
    if not messages:
        return "No previous conversation."
    lines = []
    for msg in messages[-6:]:  # last 3 exchanges only
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar — clear conversation button
with st.sidebar:
    st.markdown("### Controls")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("### Documents indexed")
    try:
        col = Chroma(
            persist_directory="vectorstore",
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        ).get()
        filenames = list(set(
            m.get("filename", "unknown")
            for m in col["metadatas"]
        ))
        for f in sorted(filenames):
            st.caption(f"📄 {f}")
    except:
        st.caption("No documents loaded")

# Render history
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
            # Build context and history
            docs = retriever.invoke(question)
            context = format_docs(docs)
            history = format_history(st.session_state.messages[:-1])

            # Run chain
            chain = (
                prompt
                | llm
                | StrOutputParser()
            )
            answer = chain.invoke({
                "context": context,
                "history": history,
                "question": question
            })

        st.write(answer)

        with st.expander("📎 Sources"):
            for doc in docs:
                st.caption(
                    f"📄 {doc.metadata.get('filename', 'unknown')} "
                    f"— page {doc.metadata.get('page', '?')+1}"
                )
                st.text(doc.page_content[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})