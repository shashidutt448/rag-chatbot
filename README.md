# RAG Chatbot – Amlgo Labs Assignment

An AI-powered chatbot built using RAG (Retrieval-Augmented Generation) capable of answering user queries from legal/contract documents with real-time responses.

## 📁 Project Structure

- `app.py` – Streamlit UI with streaming support
- `/src/` – Core logic: chunking, embedding, RAG
- `/data/` – Upload your document
- `/chunks/` – Stores chunked text
- `/vectordb/` – Saves FAISS index

## 🛠 How to Run

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## 🤖 LLM & Embedding

- LLM: Zephyr-7B
- Embeddings: all-MiniLM-L6-v2
- Vector DB: FAISS

## 🧪 Sample Queries

- "What is the refund policy?"
- "How is personal data processed?"
