import faiss
import numpy as np
import pickle
import os

embeddings_folder = "LoomRAG/embeddings"

# Load embeddings
embeddings_data = np.load(os.path.join(embeddings_folder, "clip_embeddings.npy"), allow_pickle=True).item()

# Extract filenames and embedding vectors
filenames = list(embeddings_data.keys())
embedding_vectors = np.array([embeddings_data[f][0] for f in filenames]).astype(np.float32)

print(f"Building FAISS index for {len(filenames)} images...")
print(f"Embedding dimension: {embedding_vectors.shape[1]}")

# Create FAISS index
index = faiss.IndexFlatL2(embedding_vectors.shape[1])  # L2 distance
index.add(embedding_vectors)

# Save index + filenames mapping
faiss.write_index(index, os.path.join(embeddings_folder, "clip_index.faiss"))
with open(os.path.join(embeddings_folder, "filenames.pkl"), "wb") as f:
    pickle.dump(filenames, f)

print(f"✅ FAISS index created!")
print(f"Saved to: {embeddings_folder}/clip_index.faiss")
