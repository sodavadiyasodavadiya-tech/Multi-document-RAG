from sentence_transformers import SentenceTransformer

# good accuracy + manageable speed
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_texts(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()
