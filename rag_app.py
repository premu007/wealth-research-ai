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
st.title("📄 Wealth Research AI")
st.subheader("Intelligent document analysis for financial research")

# ------------------- GROQ CLIENT -------------------
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# ------------------- EMBEDDINGS -------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ------------------- PDF LOADER -------------------
def extract_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
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

# ------------------- RETRIEVAL WITH SCORES -------------------
def retrieve_docs(vectorstore, query):
    return vectorstore.similarity_search_with_relevance_scores(query, k=4)

# ------------------- ANALYST PROMPT -------------------
def generate_answer(query, docs_with_scores):
    context = "\n\n".join([doc.page_content for doc, score in docs_with_scores])
    avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
    avg_score = round(min(max(avg_score, 0), 1), 2)

    prompt = f"""
You are a professional financial research analyst.

Your task is to analyze the provided document context and answer the query
with structured, decision-oriented output.

STRICT RULES:
- Use ONLY the provided context below
- Do NOT hallucinate or use outside knowledge
- If the answer is not in the context, say: "I couldn't find that in the document."
- Be precise and concise — this is for professional use

CONTEXT:
{context}

QUESTION:
{query}

Provide output STRICTLY in this format:

📊 Key Findings:
- Bullet points summarizing key information relevant to the question

📈 Opportunities / Positives:
- Bullet points on strengths or opportunities mentioned

⚠️ Risks / Concerns:
- Bullet points on risks, red flags, or concerns

🧠 AI Interpretation:
- What this means in practical terms for a decision-maker

🎯 Final Insight:
- One clear, actionable takeaway

Keep it precise, structured, and decision-oriented.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content, avg_score

# ------------------- UI -------------------
uploaded_file = st.file_uploader("Upload a financial PDF", type="pdf")

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    with st.spinner("Reading document..."):
        text = extract_text(uploaded_file)

    if not text.strip():
        st.error("❌ No text found. This may be a scanned PDF — try a text-based PDF.")
        st.stop()

    with st.expander("📑 Document preview (first 1000 chars)"):
        st.text(text[:1000])

    if "vectorstore" not in st.session_state or \
       st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Building knowledge base... (first time ~30 seconds)"):
            embeddings = load_embeddings()
            chunks = split_text(text)
            st.session_state.vectorstore = create_vector_store(chunks, embeddings)
            st.session_state.last_file = uploaded_file.name
            st.session_state.chunk_count = len(chunks)

    st.info(f"📚 Knowledge base ready — {st.session_state.chunk_count} chunks indexed")

    st.write("### 🔍 Ask a question")
    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g. What are the key risks? Summarize the financial outlook."
    )

    if query:
        col1, col2 = st.columns([2, 1])

        with col1:
            with st.spinner("Searching document..."):
                docs_with_scores = retrieve_docs(st.session_state.vectorstore, query)

            with st.spinner("Analysing with Llama 3.3..."):
                answer, relevance_score = generate_answer(query, docs_with_scores)

            st.write("### 🤖 AI Analysis")
            st.markdown(answer)

            st.divider()
            score_pct = int(relevance_score * 100)
            st.progress(relevance_score)
            if score_pct > 70:
                confidence = "High confidence"
            elif score_pct > 40:
                confidence = "Medium confidence"
            else:
                confidence = "Low — question may be outside document scope"
            st.caption(
                f"📌 Based only on uploaded document · "
                f"Document relevance: {score_pct}% · {confidence}"
            )

        with col2:
            with st.expander("🔍 Retrieved chunks"):
                for i, (doc, score) in enumerate(docs_with_scores):
                    st.caption(f"Chunk {i+1} · relevance: {round(score*100)}%")
                    st.write(doc.page_content[:400])
                    st.divider()

# ------------------- FOOTER -------------------
st.write("---")
st.caption("Built by Prem | Wealth Research AI · RAG System · Powered by Groq + Llama 3.3 + HuggingFace")
