from sentence_transformers import SentenceTransformer
import numpy as np
import os
import subprocess

os.makedirs("data", exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
search = [input("Search: ")]
query_vector = model.encode(search, normalize_embeddings=True).astype(np.float32)

with open("data/query.bin", "wb") as query:
    rows = query_vector.shape[0]
    cols = query_vector.shape[1]
    header = np.array([rows, cols], dtype=np.int32)
    query.write(header.tobytes())
    query.write(query_vector.tobytes())

main_process = subprocess.run(["./main"], capture_output=True, text=True)
print(main_process.stdout)