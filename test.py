import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
chunks_file = "chunks_semantic.json"
faiss_index_file = "faiss_indices/semantic_chunks.faiss"

# Load clean Q–A
with open(chunks_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Prepare texts for embedding (combine Q + A)
texts = [f"Q: {c['question']} A: {c['answer']}" for c in chunks]

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity (inner product)
index.add(embeddings)

# Save index
faiss.write_index(index, faiss_index_file)
print(f"✅ Saved FAISS index with {len(chunks)} entries at {faiss_index_file}")
