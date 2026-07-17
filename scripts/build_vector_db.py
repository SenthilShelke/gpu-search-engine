from sentence_transformers import SentenceTransformer
import numpy as np
import os

import corpus

MAX_ROWS = 50000
BATCH_SIZE = 1000


def write_vector_db(path: str, matrix: np.ndarray) -> None:
    """
    Write a matrix to the Vector_Db_File format: an 8-byte header of two
    little-endian int32 values (row count, column count) followed by the
    matrix values as little-endian float32 in row-major order.
    """
    matrix = np.asarray(matrix, dtype="<f4")
    rows, cols = matrix.shape
    header = np.array([rows, cols], dtype="<i4")

    with open(path, "wb") as db:
        db.write(header.tobytes())
        db.write(np.ascontiguousarray(matrix).tobytes())


def read_vector_db(path: str) -> np.ndarray:
    """
    Read a Vector_Db_File and return its contents as a (rows, cols) float32
    matrix. Inverse of write_vector_db.
    """
    with open(path, "rb") as db:
        header = np.frombuffer(db.read(8), dtype="<i4")
        rows, cols = int(header[0]), int(header[1])
        data = np.frombuffer(db.read(rows * cols * 4), dtype="<f4")
        return data.reshape((rows, cols))

os.makedirs("data", exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
corpus_list = corpus.get_corpus(max_rows=MAX_ROWS)

cols = 384
batches = []
for i in range(0, len(corpus_list), BATCH_SIZE):
    sentences = corpus_list[i : i + BATCH_SIZE]
    embedded_vectors = model.encode(sentences, normalize_embeddings=True).astype(np.float32)
    print(f"Embedding sentences {i} to {i + len(sentences)}")
    batches.append(embedded_vectors)

embeddings = np.concatenate(batches, axis=0) if batches else np.empty((0, cols), dtype=np.float32)
write_vector_db("data/vector_db.bin", embeddings)