from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import os
import subprocess
import time

os.makedirs("data", exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dataset = load_dataset("SetFit/ag_news", split="train")

main_process = subprocess.Popen(["./main"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

while True:
    search = [input("Search: ")]
    if search == ["EXIT"]:
        break
    query_vector = model.encode(search, normalize_embeddings=True).astype(np.float32)

    start = time.perf_counter()

    main_process.stdin.write(query_vector.tobytes())
    main_process.stdin.flush()
    byte_results = main_process.stdout.read(20)

    end = time.perf_counter()

    top_indices = np.frombuffer(byte_results, dtype=np.int32)
    print(f"\nLatency: {(end - start) * 1000:.2f} ms\n")
    for i in range(5):
        print(dataset[top_indices[i]]["text"])
