import tiktoken


def chunk_text(text: str, chunk_size=500, overlap=80):
    """
    Token based chunking = best for accuracy because models work on tokens.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        start += (chunk_size - overlap)

    return chunks
