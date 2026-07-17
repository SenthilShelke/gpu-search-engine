from sentence_transformers import SentenceTransformer
import numpy as np
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import corpus

os.makedirs("data", exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
corpus_list = corpus.get_corpus(max_rows=50000)

main_process = subprocess.Popen(["./main"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

SANITY_CHECK_INDEX = 0
sanity_query_vector = model.encode(
    [corpus_list[SANITY_CHECK_INDEX]], normalize_embeddings=True
).astype(np.float32)
main_process.stdin.write(sanity_query_vector.tobytes())
main_process.stdin.flush()
sanity_byte_results = main_process.stdout.read(20)
sanity_top_indices = np.frombuffer(sanity_byte_results, dtype=np.int32)
if SANITY_CHECK_INDEX not in sanity_top_indices:
    print(
        f"WARNING: self-retrieval sanity check failed - expected index "
        f"{SANITY_CHECK_INDEX} not found in top-5 results {list(sanity_top_indices)}."
    )

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
        print(corpus_list[top_indices[i]])
