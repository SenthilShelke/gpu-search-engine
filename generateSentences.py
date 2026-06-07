from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dataset = load_dataset("SetFit/ag_news", split="train")
sentences = []

for i in range(10):
    sentences.append(dataset[i]["text"])
    
embeddedVectors = model.encode(sentences)

print(embeddedVectors)