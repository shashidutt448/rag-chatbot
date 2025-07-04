import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.utils import load_document, clean_text, split_text
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


st.set_page_config(layout="wide", page_title=" RAG Chatbot | Amlgo Labs")
st.title("🤖 AI-Powered Legal Document Chatbot")

# Sidebar
st.sidebar.markdown("### 🔍 Model Details")
st.sidebar.markdown("""
**LLM:** Zephyr-7B  
**Embedding:** all-MiniLM-L6-v2  
**Vector DB:** FAISS
""")

# Local model path
MODEL_ID = "./models/zephyr-7b"

# File Upload
uploaded_file = st.file_uploader("📎 Upload Document (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔄 Reading and processing the document..."):
        try:
            raw_text = load_document(uploaded_file)
            cleaned = clean_text(raw_text)
            chunks = split_text(cleaned)

            if not chunks:
                st.error(" No readable text found in the uploaded PDF.")
                st.stop()

            rag = RAGPipeline(chunks, model_id=MODEL_ID)
            st.success(f" Document processed into {len(chunks)} chunks.")
        except Exception as e:
            st.error(f" Failed during document processing or model loading:\n\n{e}")
            st.stop()

    # User Query
    query = st.text_input(" Ask something about the document:")
    if query:
        with st.spinner("💬 Generating response..."):
            try:
                answer, source_chunks = rag.query(query)
                st.markdown("### 📌 **Answer**")
                st.markdown(f"{answer}")

                with st.expander(" Source Chunks"):
                    for i, chunk in enumerate(source_chunks):
                        st.markdown(f"**Chunk {i+1}:**\n> {chunk}")
            except Exception as e:
                st.error(f" Error during query execution:\n\n{e}")
