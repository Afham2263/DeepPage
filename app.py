import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
import os
import torch
from dotenv import load_dotenv

# Load .env for local dev
load_dotenv()

# Try to get key from .env or Streamlit Cloud Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.error("No GROQ_API_KEY found. Set it in a .env file (local) or Streamlit Secrets (cloud).")
    st.stop()

# Load embeddings + model with meta tensor fix
try:
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except RuntimeError as e:
        if "meta tensor" in str(e):
            torch.nn.Module.to_empty = lambda self, *args, **kwargs: self
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        else:
            raise
except Exception as e:
    st.error(f"Error loading embedding model: {e}")
    st.stop()

try:
    index = faiss.read_index("embeddings/faiss_index.bin")
    docs, sources = pickle.load(open("embeddings/metadata.pkl", "rb"))
except FileNotFoundError:
    st.error("FAISS index or metadata.pkl not found. Run your ingestion script first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading index or metadata: {e}")
    st.stop()

# Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

def search(query, k=3):
    try:
        q_vec = model.encode([query])
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, k)
        return [(docs[i], sources[i]) for i in I[0]]
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def answer_with_groq(question, context):
    try:
        prompt = f"Answer the question using only the context below. If it is not in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"

# Streamlit UI
st.title("AskDocs - Wikipedia Q&A")
question = st.text_input("Ask a question about your document:")

if st.button("Search & Answer"):
    if not question.strip():
        st.warning("Please enter a valid question before searching.")
    else:
        matches = search(question)
        if not matches:
            st.warning("No results found for your query.")
        else:
            context = "\n".join([m[0] for m in matches])
            answer = answer_with_groq(question, context)
            st.subheader("Answer")
            st.write(answer)

            with st.expander("Sources Used"):
                for chunk, src in matches:
                    st.write(f"From {src}:\n{chunk}")
