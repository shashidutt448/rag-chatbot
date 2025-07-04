import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        print("Embedding", len(texts), "chunks")
        if not texts or not isinstance(texts, list) or len(texts[0].strip()) == 0:
            raise ValueError(" No valid text chunks to embed.")
        return self.model.encode(texts, convert_to_numpy=True)


    def build_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
