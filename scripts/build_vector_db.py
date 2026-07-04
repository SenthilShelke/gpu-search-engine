from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import os

TOTAL_SENTENCES = 50000
BATCH_SIZE = 1000

os.makedirs("data", exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dataset = load_dataset("SetFit/ag_news", split="train")

with open("data/vector_db.bin", "wb") as db:
    rows = TOTAL_SENTENCES
    cols = 384
    header = np.array([rows, cols], dtype=np.int32)
    db.write(header.tobytes())

    for i in range(0, TOTAL_SENTENCES, BATCH_SIZE):
        sentences = dataset[i : i + BATCH_SIZE]["text"]
        embedded_vectors = model.encode(sentences, normalize_embeddings=True).astype(np.float32)
        db.write(embedded_vectors.tobytes())