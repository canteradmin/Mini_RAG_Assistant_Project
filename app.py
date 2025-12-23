
import re
from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from prompts import build_messages, MAX_HISTORY_TURNS
# Load environment variables
#load_dotenv(dotenv_path="RAG.env")
OPENAI_API_KEY = "your-api-key"  
client = OpenAI(api_key=OPENAI_API_KEY)
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "rag-documents")
DATA_DIR = Path("knowledgebase")
VECTORSTORE_DIR = Path("knowledgebase_vectorstore")
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
# HELPERS
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)

def get_local_loader(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    if ext in [".docx", ".doc"]:
        return Docx2txtLoader(str(file_path))
    if ext == ".txt":
        return TextLoader(str(file_path), encoding="utf-8")
    raise ValueError("Unsupported file type")

@st.cache_data
def load_and_split_local_documents():
    docs = []
    for file_path in DATA_DIR.glob("*.*"):
        try:
            loader = get_local_loader(file_path)
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = file_path.name
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"Failed to load {file_path.name}: {e}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

@st.cache_resource
def load_vectorstore_local():
    if any(VECTORSTORE_DIR.iterdir()):
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    chunks = load_and_split_local_documents()
    if not chunks:
        return None
    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(str(VECTORSTORE_DIR))
    return vs

@st.cache_resource
def load_vectorstore_azure():
    if not AZURE_CONNECTION_STRING:
        st.error("Azure connection string not set.")
        return None
    try:
        loader = AzureBlobStorageContainerLoader(
            conn_str=AZURE_CONNECTION_STRING,
            container=AZURE_CONTAINER_NAME
        )
        raw_docs = loader.load()
        if not raw_docs:
            st.warning(f"No documents in container '{AZURE_CONTAINER_NAME}'")
            return None
        for doc in raw_docs:
            doc.metadata["source"] = f"azure://{AZURE_CONTAINER_NAME}/{doc.metadata.get('name', 'unknown')}"
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(raw_docs)
        return FAISS.from_documents(chunks, get_embeddings())
    except Exception as e:
        st.error(f"Azure load failed: {e}")
        return None

# METRICS (unchanged)

def precision_at_k(docs, query, k=3):
    q_words = set(query.lower().split())
    hits = sum(1 for d in docs[:k] if any(w in d.page_content.lower() for w in q_words))
    return hits / k if k else 0.0

def hit_rate_at_k(docs, answer, k=3):
    keywords = [w for w in re.findall(r"\w+", answer.lower()) if len(w) > 3]
    for d in docs[:k]:
        if any(w in d.page_content.lower() for w in keywords):
            return 1.0
    return 0.0

def faithfulness(answer, context):
    aw = set(re.findall(r"\w+", answer.lower()))
    cw = set(re.findall(r"\w+", context.lower()))
    return len(aw & cw) / max(len(aw), 1)

def compute_confidence(p, h, f):
    return round((0.4 * p) + (0.3 * h) + (0.3 * f), 2)

# STREAMLIT UI
st.set_page_config("Mini RAG Assistant", layout="wide")

st.markdown("""
<style>
.stApp { background:#000; color:#fff; }
.answer-box { background:#111; border-left:4px solid #ffc107; padding:16px; margin:12px 0; }
.highlight-answer { background-color:#fff3a0 !important; border:2px solid #ff9800 !important; padding:12px; border-radius:6px; color:#000 !important; white-space:pre-wrap; }
.normal-chunk { border:1px solid #333; padding:12px; margin:8px 0; white-space:pre-wrap; }
.stButton>button { background:#ffc107; color:#000; }
</style>
""", unsafe_allow_html=True)

st.title("Mini RAG Assistant")

# SIDEBAR
with st.sidebar:
    st.header("Data Source")
    data_source = st.radio(
        "Choose source",
        options=["Local", "Cloud (Azure Blob Storage)"],
        index=0
    )
    USE_CLOUD = (data_source == "Cloud (Azure Blob Storage)")

    if USE_CLOUD:
        st.info(f"Using Azure container: **{AZURE_CONTAINER_NAME}**")
        st.caption("Documents loaded from cloud storage.")

    st.markdown("---")

    if not USE_CLOUD:
        st.header("Upload Document")
        uploaded = st.file_uploader("Upload pdf, docx, txt", type=["pdf", "docx", "txt"])
        if uploaded:
            with open(DATA_DIR / uploaded.name, "wb") as f:
                f.write(uploaded.getbuffer())
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success(f"Uploaded {uploaded.name}")

# SESSION STATE
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_source" not in st.session_state:
    st.session_state.last_source = None

# QUERY INPUT
query = st.text_input("Ask a question...")

if st.button("Send") and query:
    current_source = "cloud" if USE_CLOUD else "local"

    if (st.session_state.vectorstore is None or
        st.session_state.last_source != current_source):

        with st.spinner("Loading knowledge base..."):
            if USE_CLOUD:
                st.session_state.vectorstore = load_vectorstore_azure()
            else:
                st.session_state.vectorstore = load_vectorstore_local()
            st.session_state.last_source = current_source

    if st.session_state.vectorstore is None:
        st.error("No documents loaded. Upload files (Local) or check Azure config (Cloud).")
    else:
        docs = st.session_state.vectorstore.similarity_search(query, k=3)
        context = "\n".join(d.page_content for d in docs)

        messages = build_messages(query, context, st.session_state.conversation)

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()

        p = precision_at_k(docs, query)
        h = hit_rate_at_k(docs, answer)
        f = faithfulness(answer, context)
        confidence = compute_confidence(p, h, f)

        st.session_state.conversation.append({
            "query": query,
            "answer": answer,
            "docs": docs,
            "context": context,
            "metrics": {
                "precision": p,
                "hit_rate": h,
                "faithfulness": f,
                "confidence": confidence
            }
        })

# DISPLAY CONVERSATION
for turn in reversed(st.session_state.conversation):
    st.markdown(f"**{turn['query']}**")
    st.markdown(f'<div class="answer-box">{turn["answer"]}</div>', unsafe_allow_html=True)

    with st.expander("Metrics"):
        m = turn["metrics"]
        st.write(f"Precision@3: {m['precision']:.2f}")
        st.write(f"Hit Rate@3: {m['hit_rate']:.2f}")
        st.write(f"Faithfulness: {m['faithfulness']:.2f}")
        st.write(f"Confidence Score: {m['confidence']:.2f}")

    answer_text = turn["answer"].lower()
    keywords = {w for w in re.findall(r"\w+", answer_text) if len(w) > 4}

    best_idx = -1
    max_matches = 0
    for idx, doc in enumerate(turn["docs"]):
        matches = sum(1 for w in keywords if w in doc.page_content.lower())
        if matches > max_matches:
            max_matches = matches
            best_idx = idx

    with st.expander("Retrieved Chunks"):
        for idx, doc in enumerate(turn["docs"]):
            if idx == best_idx and max_matches >= 2:
                st.markdown(f'<div class="highlight-answer">{doc.page_content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="normal-chunk">{doc.page_content}</div>', unsafe_allow_html=True)