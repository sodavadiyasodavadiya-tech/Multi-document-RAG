import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi


class VectorStore:
    def __init__(self, dim: int, index_path="data/index/faiss.index", meta_path="data/index/meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self.bm25 = None
        self.bm25_corpus = []

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load()

    def add(self, embeddings, metadatas):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

        # update BM25 corpus
        for md in metadatas:
            self.bm25_corpus.append(md["text"].split())

        self.bm25 = BM25Okapi(self.bm25_corpus)
        self.save()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.bm25_corpus = [m["text"].split() for m in self.metadata]
        self.bm25 = BM25Okapi(self.bm25_corpus) if self.bm25_corpus else None

    def search_dense(self, query_embedding, top_k=10):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.metadata[idx])
        return results

    def search_bm25(self, query, top_k=10):
        if not self.bm25:
            return []
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(self.metadata[idx])
        return results

    def hybrid_search(self, query_embedding, query_text, top_k=10):
        dense_results = self.search_dense(query_embedding, top_k=top_k)
        bm25_results = self.search_bm25(query_text, top_k=top_k)

        # merge unique by chunk id
        seen = set()
        merged = []
        for r in dense_results + bm25_results:
            cid = r["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                merged.append(r)

        return merged[:top_k]
    
    def reset(self):
        """
        Reset the vector database by clearing all data and deleting index files.
        This will remove all indexed documents and their embeddings.
        """
        # Reset in-memory data
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []
        self.bm25 = None
        self.bm25_corpus = []
        
        # Delete index files if they exist
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)

