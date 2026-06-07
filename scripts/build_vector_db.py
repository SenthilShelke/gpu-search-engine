from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import os

os.makedirs("data", exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dataset = load_dataset("SetFit/ag_news", split="train")
sentences = []

for i in range(10):
    sentences.append(dataset[i]["text"])
    
embedded_vectors = model.encode(sentences, normalize_embeddings=True).astype(np.float32)

with open("data/vector_db.bin", "wb") as db:
    rows = embedded_vectors.shape[0]
    cols = embedded_vectors.shape[1]
    header = np.array([rows, cols], dtype=np.int32)
    db.write(header.tobytes())
    db.write(embedded_vectors.tobytes())