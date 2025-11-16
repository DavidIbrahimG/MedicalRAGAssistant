import os
import time
import streamlit as st
from dotenv import load_dotenv
import tempfile

# LangChain / Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# RAG stack
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# ----------------------
# ENV & PAGE CONFIG
# ----------------------
load_dotenv()
st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------
# SESSION STATE INIT
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant", "content": "..."}]

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "embed_stats" not in st.session_state:
    st.session_state.embed_stats = {"docs": 0, "chunks": 0}

# ----------------------
# LLM FACTORY
# ----------------------
def make_llm(model_name: str, temperature: float):
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Add it to your environment or .streamlit/secrets.toml")
        st.stop()
    return ChatGroq(model=model_name, api_key=GROQ_API_KEY, temperature=temperature)

# ----------------------
# PROMPT
# ----------------------
PROMPT = ChatPromptTemplate.from_template(
    """You are a medical research assistant.
Answer strictly from the provided context. If the answer isn't in the context, say you don't have enough information.
Give a concise, accurate answer and expand with brief reasoning or suggested next steps if applicable.

<context>
{context}
</context>

Question: {input}"""
)

# ----------------------
# BUILD / REFRESH INDEX
# ----------------------
def build_index(base_dir: str, uploaded_files, chunk_size: int, chunk_overlap: int):
    docs = []

    # Load PDFs from a folder (optional)
    if base_dir and os.path.isdir(base_dir):
        loader = PyPDFDirectoryLoader(base_dir)
        docs.extend(loader.load())

    # Load uploaded PDFs safely (no st.secrets needed)
    if uploaded_files:
        for up in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up.read())
                tmp_path = tmp.name
            docs.extend(PyPDFLoader(tmp_path).load())

    if not docs:
        st.warning("No documents found. Add PDFs in ./data or upload from the sidebar.")
        return None, 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = FAISS.from_documents(splits, embeddings)
    return vectors, len(docs), len(splits)


# ----------------------
# UI - SIDEBAR
# ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Groq model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    k_neighbors = st.slider("Top-K documents", 1, 10, 4)

    st.markdown("---")
    st.subheader("📄 Documents")
    uploaded = st.file_uploader(
        "Upload PDFs (optional)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can also place PDFs in ./data",
    )
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, 10)

    col_a, col_b = st.columns(2)
    with col_a:
        do_embed = st.button("🔎 Build / Refresh Index")
    with col_b:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

    if do_embed:
        with st.spinner("Building vector index…"):
            start = time.time()
            vectors, ndocs, nchunks = build_index(
                base_dir="data",
                uploaded_files=uploaded,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            elapsed = time.time() - start
            if vectors:
                st.session_state.vectors = vectors
                st.session_state.embed_stats = {"docs": ndocs, "chunks": nchunks}
                st.success(f"Index ready • {ndocs} docs → {nchunks} chunks • {elapsed:.2f}s")

    if st.session_state.vectors:
        st.caption(
            f"✅ Index loaded • {st.session_state.embed_stats['docs']} docs / "
            f"{st.session_state.embed_stats['chunks']} chunks"
        )
    else:
        st.caption("❌ No index loaded")

# ----------------------
# MAIN HEADER
# ----------------------
st.title("🩺 Medical Assistant")
st.markdown(
    "Ask questions about your uploaded papers or PDFs in `./data`. "
    "Use the sidebar to upload files and build the index."
)

# ----------------------
# CHAT HISTORY (DISPLAY)
# ----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Optional: show sources for previous assistant messages
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"- **{i}.** {s}")

# ----------------------
# CHAT INPUT
# ----------------------
user_msg = st.chat_input("Type your question about the papers…")

if user_msg:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    if not st.session_state.vectors:
        with st.chat_message("assistant"):
            st.warning("Please build the index first from the sidebar (🔎 Build / Refresh Index).")
    else:
        # Build chain
        llm = make_llm(model_name, temperature)
        document_chain = create_stuff_documents_chain(llm, PROMPT)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": k_neighbors})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                t0 = time.time()
                result = retrieval_chain.invoke({"input": user_msg})
                dt = time.time() - t0

                # Write answer
                answer = result.get("answer") or result.get("output_text") or "I couldn't find enough info in the context."
                st.write(answer)
                st.caption(f"⏱️ {dt:.2f}s • k={k_neighbors} • model={model_name}")

                # Prepare/Show sources
                sources_md = []
                ctx_docs = result.get("context", []) or result.get("documents", [])
                if ctx_docs:
                    with st.expander("Sources"):
                        for i, d in enumerate(ctx_docs, 1):
                            # Try to show filename + page if present
                            meta = d.metadata or {}
                            name = meta.get("source") or meta.get("file_path") or "Document"
                            page = meta.get("page", None)
                            label = f"{name}" + (f" — page {page}" if page is not None else "")
                            sources_md.append(label)
                            st.markdown(f"- **{i}.** {label}")

        # Save assistant message + sources into history
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources_md}
        )

