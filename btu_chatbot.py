
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from gpt4all import GPT4All

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_index.idx")
    with open("chunked_documents.pkl", "rb") as f:
        chunks = pickle.load(f)
    model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", model_path="models", allow_download=False)
    return embedder, index, chunks, model

embedding_model, index, chunked_documents, model = load_resources()

def answer_btu_query(query: str, k: int = 2, max_tokens: int = 250, temp: float = 0.3) -> str:
    try:
        query_embedding = embedding_model.encode([query])
        D, I = index.search(np.array(query_embedding), k)
        retrieved_chunks = [chunked_documents[i][:1200] for i in I[0]]
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""You are a helpful assistant for a team learning about BTU functions and product management.
Use only the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""
        response = model.generate(prompt, max_tokens=max_tokens, temp=temp)
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

st.set_page_config(page_title="BTU Chatbot", page_icon="🤖")
st.title("🤖 BTU Chatbot")
st.write("Ask me anything about BTU functions and product management.")

query = st.text_input("Enter your question:", placeholder="e.g. What systems does the BTU team support?")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            response = answer_btu_query(query)
        st.success("Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question.")
