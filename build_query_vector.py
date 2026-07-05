from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import os
import subprocess
import time

os.makedirs("data", exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dataset = load_dataset("SetFit/ag_news", split="train")
search = [input("Search: ")]
query_vector = model.encode(search, normalize_embeddings=True).astype(np.float32)

with open("data/query.bin", "wb") as query:
    rows = query_vector.shape[0]
    cols = query_vector.shape[1]
    header = np.array([rows, cols], dtype=np.int32)
    query.write(header.tobytes())
    query.write(query_vector.tobytes())

start = time.perf_counter()
subprocess.run(["./main"])
end = time.perf_counter()

top_indices = np.fromfile("data/results.bin", dtype=np.int32)
for i in range(5):
    print(dataset[top_indices[i]]["text"])

print(f"\nLatency: {(end - start) * 1000:.2f} ms")