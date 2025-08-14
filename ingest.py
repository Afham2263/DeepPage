import os
import pickle
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def pdf_to_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

docs = []
sources = []

# Load and process PDFs
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        text = pdf_to_text(os.path.join("data", file))
        chunks = chunk_text(text)
        docs.extend(chunks)
        sources.extend([file] * len(chunks))

if not docs:
    raise ValueError("No PDFs found in data/ folder.")

# Create embeddings
embeddings = model.encode(docs)

# Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index & metadata
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/faiss_index.bin")
with open("embeddings/metadata.pkl", "wb") as f:
    pickle.dump((docs, sources), f)

print(f"Indexed {len(docs)} chunks from {len(set(sources))} documents.")
