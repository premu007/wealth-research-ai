import streamlit as st
import os
from groq import Groq
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Wealth Research AI", layout="wide")
st.title("📄 Wealth Research AI (RAG)")
st.subheader("Upload a financial document and ask intelligent questions")

# ------------------- GROQ CLIENT -------------------
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# ------------------- EMBEDDINGS (free, local) -------------------
# Loaded once and cached — avoids reloading on every Streamlit rerun
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ------------------- PDF LOADER -------------------
# FIX vs skeleton: handles None from scanned/image pages safely
def extract_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:                          # skip pages with no text
            text += extracted + "\n"
    return text

# ------------------- TEXT SPLITTING -------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# ------------------- VECTOR STORE -------------------
def create_vector_store(chunks, embeddings):
    docs = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(docs, embeddings)

# ------------------- RETRIEVAL -------------------
def retrieve_docs(vectorstore, query):
    return vectorstore.similarity_search(query, k=4)

# ------------------- LLM ANSWER -------------------
def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a professional financial research assistant.
Answer ONLY using the context provided below.
If the answer is not in the context, say: "I couldn't find that in the document."

CONTEXT:
{context}

QUESTION:
{query}

Provide a clear, structured answer with bullet points where appropriate.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3                        # low = more factual, less creative
    )
    return response.choices[0].message.content

# ------------------- UI -------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Step 1 — Extract
    with st.spinner("Reading document..."):
        text = extract_text(uploaded_file)

    if not text.strip():
        st.error("❌ No text found. This may be a scanned PDF — try a text-based PDF.")
        st.stop()

    # Step 2 — Preview
    with st.expander("📑 Document preview (first 1000 chars)"):
        st.text(text[:1000])

    # Step 3 — Chunk + embed + store (only once per file upload)
    if "vectorstore" not in st.session_state or \
       st.session_state.get("last_file") != uploaded_file.name:

        with st.spinner("Building knowledge base... (first time takes ~30 seconds)"):
            embeddings = load_embeddings()
            chunks = split_text(text)
            st.session_state.vectorstore = create_vector_store(chunks, embeddings)
            st.session_state.last_file = uploaded_file.name
            st.session_state.chunk_count = len(chunks)

    st.info(f"📚 Knowledge base ready — {st.session_state.chunk_count} chunks indexed")

    # Step 4 — Query
    st.write("### 🔍 Ask a question")
    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g. What are the key risks mentioned? Summarize the main findings."
    )

    if query:
        col1, col2 = st.columns([2, 1])

        with col1:
            with st.spinner("Searching document..."):
                docs = retrieve_docs(st.session_state.vectorstore, query)

            with st.spinner("Generating answer..."):
                answer = generate_answer(query, docs)

            st.write("### 🤖 Answer")
            st.markdown(answer)

        with col2:
            with st.expander("🔍 Retrieved chunks (debug)"):
                for i, doc in enumerate(docs):
                    st.caption(f"Chunk {i+1}")
                    st.write(doc.page_content[:400])
                    st.divider()

# ------------------- FOOTER -------------------
st.write("---")
st.caption("Built by Prem.C| Wealth Research AI | RAG System | Powered by Groq + Llama 3.3 + HuggingFace")
