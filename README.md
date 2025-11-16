
# 🩺 Medical RAG Assistant (Streamlit + LangChain + Groq + FAISS)

A production-ready **Retrieval-Augmented Generation (RAG)** app for querying medical research papers.  
It combines **Groq LLMs** (Llama 3 family), **LangChain** orchestration, **FAISS** vector search, and a clean **Streamlit** chat UI with conversation history, file uploads, and source citations.

---

## ✨ What This Project Does

- **Upload PDFs** (or place them in `./data/`) and build a semantic index.
- Ask questions in a **chat interface** powered by Groq LLMs.
- Retrieves relevant chunks via **FAISS** and composes answers with **LangChain**.
- **Cites sources** (file name + page) for transparency.
- Adjustable **chunking**, **Top‑K**, **temperature**, and **model** from the sidebar.
- Includes **conversation history** in-session and a **Clear Chat** button.

---
## App UI

<img width="1459" height="875" alt="Screenshot 2025-11-16 at 21 16 56" src="https://github.com/user-attachments/assets/f9eabd1e-2bb4-4c06-943c-a51507d94a42" />


<img width="1464" height="878" alt="Screenshot 2025-11-16 at 21 19 21" src="https://github.com/user-attachments/assets/93587b07-d4f7-4390-b990-ee41bedaea15" />


---

## 🧰 Tech Stack

- **Frontend / UX:** Streamlit (chat UI: `st.chat_message`, sidebar controls)
- **Orchestration:** LangChain (prompting, document chains, retrieval)
- **LLMs:** Groq (`llama-3.1-8b-instant`, `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`)
- **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
- **Vector Store / Retriever:** FAISS
- **Loaders:** `PyPDFDirectoryLoader`, `PyPDFLoader`
- **Env Management:** `python-dotenv`

---

## 🏗️ Architecture Overview

```
PDFs (./data + uploads) 
      └── PyPDF loaders → Text Splitter (RecursiveCharacterTextSplitter)
             └── Embeddings (HF MiniLM) → Vector Store (FAISS)
                    └── Retriever (Top-K, metadata filtering)
                           └── Stuff Documents Chain (LangChain)
                                  └── Groq LLM (Llama/Mixtral)
                                         └── Chat UI (Streamlit) + Sources + History
```

- **Index lifecycle:** Clicking **Build / Refresh Index** loads *all* PDFs from `./data` **plus** any uploaded files and **rebuilds** the index.
- **Uploads** are combined **in-memory** with `./data` files for that run. Restarting requires re-upload (unless persisted; see below).

---

## 🆚 Comparative Choices & Why

### LLMs
| Option | Pros | Cons | When to Use |
|---|---|---|---|
| **Groq (Llama 3.x, Mixtral)** | Very low latency, cost-effective, strong 70B quality | API dependency | Fast UX, cost/latency sensitive apps |
| **OpenAI (GPT-4o/4.1)** | Top-tier quality, strong reasoning | Higher cost/latency, policy limits | Highest quality answers / evaluation |
| **Ollama (local LLMs)** | Fully local, private | GPU/CPU burden, varying model quality | Offline, POCs, privacy constraints |

*Current default:* **Groq `llama-3.1-8b-instant`** (snappy, solid quality). Switchable in the sidebar.

### Embeddings
| Option | Pros | Cons |
|---|---|---|
| **HF `all-MiniLM-L6-v2`** | Free, fast, widely used | Slightly less semantic nuance than larger models |
| OpenAI `text-embedding-3-large/small` | High quality | Paid, requires API key |
| Instructor / E5 / GTE | Domain-tuned variants available | Needs evaluation per corpus |

*Chosen for balance:* **MiniLM-L6-v2**.

### Vector Stores
| Store | Pros | Cons |
|---|---|---|
| **FAISS** | Local, fast, simple | No managed scaling, you persist manually |
| Pinecone | Managed, scales, metadata filters | Paid beyond free tier |
| Chroma | Simple dev UX, local or server | Fewer managed features |
| Astra DB (Vector) | Serverless, scalable, hybrid search | Cloud setup required |

*Default:* **FAISS** for local simplicity. Can swap to Pinecone/Chroma with minimal code changes.

### Retrieval Strategy
- **Top‑K** (default 4), **MMR** (optional), filters by source/page, chunking tuned via sidebar.
- **Stuff chain** for small context; switch to **map‑reduce** for big chunks/corpora.

---

## 📦 Project Structure

```
medgenai/
├─ app.py                       # Streamlit app (chat UI + sidebar + RAG)
├─ data/                        # Put baseline PDFs here
├─ .env                         # GROQ_API_KEY etc. (not committed)
├─ requirements.txt
└─ README.md                    # This file
```

---

## 🔑 Environment & Secrets

Create a `.env` at project root:

```env
GROQ_API_KEY=sk_your_groq_key
HUGGINGFACE_API_KEY=optional_if_needed
```

> If you use Streamlit secrets instead, create `.streamlit/secrets.toml` and set the same keys. The app also works with plain env vars.

---

## ✅ Installation & Run

```bash
# 1) Create & activate a venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate

# 2) Upgrade pip & install
python -m pip install -U pip
pip install -r requirements.txt

# 3) Set env (or use .env as above)
export GROQ_API_KEY="sk_..."

# 4) Run
python -m streamlit run app.py
```

### Suggested `requirements.txt`
```
streamlit==1.39.0
langchain==0.3.7
langchain-core==0.3.15
langchain-community==0.3.5
langchain-groq==0.2.0
groq==0.13.1
pydantic>=2,<3
faiss-cpu
python-dotenv
pypdf
```

---

## 🧩 Key Code Snippets

### Sidebar: Build / Refresh Index (Uploads + ./data)
```python
import tempfile
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_index(base_dir, uploaded_files, chunk_size, chunk_overlap):
    docs = []

    if base_dir and os.path.isdir(base_dir):
        docs.extend(PyPDFDirectoryLoader(base_dir).load())

    if uploaded_files:
        for up in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up.read())
                tmp_path = tmp.name
            docs.extend(PyPDFLoader(tmp_path).load())

    if not docs:
        return None, 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["

", "
", " ", ""],
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = FAISS.from_documents(splits, embeddings)
    return vectors, len(docs), len(splits)
```

### Chat UI + Retrieval
```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chains.combine_documents import create_stuff_documents_chain
from langchain_chains import create_retrieval_chain

PROMPT = ChatPromptTemplate.from_template(
    """You are a medical research assistant.
Answer strictly from the provided context. If unknown, say so.
<context>
{context}
</context>
Question: {input}"""
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=os.getenv("GROQ_API_KEY"))
doc_chain = create_stuff_documents_chain(llm, PROMPT)
retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
chain = create_retrieval_chain(retriever, doc_chain)
result = chain.invoke({"input": user_msg})
```

---

## 🧪 Quality, Evaluation & Tuning

- **Chunking:** start with 800–1200 chars, overlap 150–250. Increase for long-form PDFs.
- **Top‑K:** 3–6 works well; add **MMR** for diversity if answers feel narrow.
- **Prompting:** keep system prompt strict about “use only context”; add a fallback line for “insufficient info”.
- **LLM choice:** use 8B for speed, 70B for thoroughness; benchmark latency and answer quality on your corpus.

---

## 🔐 Security & Compliance

- PDFs may contain sensitive data. Consider **local-only** operation or VPC networking.
- Turn on **anonymization** / scrubbing before indexing if needed.
- Log **minimal** inputs/outputs. Avoid storing raw PHI/PII.

---

## 🚀 Deployment Options

### Local
- `python -m streamlit run app.py` (default).

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t medical-rag .
docker run -p 8501:8501 -e GROQ_API_KEY=sk_... medical-rag
```

### Render / Hugging Face Spaces / EC2
- Ensure `PORT` is respected or set `--server.port=$PORT`.
- For Render: Web Service → `python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🗃️ Persistence (optional)

Avoid re-embedding every run:
```python
# Save
st.session_state.vectors.save_local("faiss_index")

# Load
from langchain_community.vectorstores import FAISS
vectors = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

Persist chat history across sessions:
```python
import json, pathlib
HIST = pathlib.Path(".cache/history.json")

def save_hist():
    HIST.parent.mkdir(parents=True, exist_ok=True)
    HIST.write_text(json.dumps(st.session_state.messages, ensure_ascii=False))

def load_hist():
    if HIST.exists():
        st.session_state.messages = json.loads(HIST.read_text())
```

---

## 🛠️ Troubleshooting

- **`ModuleNotFoundError: No module named 'langchain_groq'`**  
  Install compatible versions:  
  `pip install langchain==0.3.7 langchain-groq==0.2.0 groq==0.13.1`

- **Secrets error (`StreamlitSecretNotFoundError`)**  
  Remove use of `st.secrets[...]` for temp file paths; use `tempfile` instead (see code above).

- **“Build index first” warning**  
  Click **Build / Refresh Index** after adding PDFs.

- **Slow responses**  
  Lower Top‑K, reduce chunk size, switch to `llama-3.1-8b-instant`, or raise temperature slightly for shorter outputs.

---

## 🧭 Roadmap

- [ ] Persist FAISS index + chat history automatically
- [ ] MMR retriever & metadata filters
- [ ] Multi-tenant namespaces / collections
- [ ] Source preview (page image snippets)
- [ ] Optional Pinecone/Chroma backend
- [ ] Basic guardrails + PII/PHI scrubbing

---

## 📜 License

MIT — see `LICENSE` (add if missing).

---

## 🤝 Contributing

PRs welcome! Please open an issue to discuss significant changes.
