import os
from document_loader import load_document
from chunker import chunk_text
from vectordb import VectorStore
from embeddings import embed_texts
from llm_router import generate_answer_with_fallback


# bge-small dimension = 384
def build_index(file_path: str, vectordb: VectorStore):
    text = load_document(file_path)
    chunks = chunk_text(text, chunk_size=500, overlap=80)

    embeddings = embed_texts(chunks)

    metadatas = []
    for i, ch in enumerate(chunks):
        metadatas.append({
            "chunk_id": f"{os.path.basename(file_path)}_{i}",
            "source": os.path.basename(file_path),
            "text": ch
        })

    vectordb.add(embeddings, metadatas)

    return {"chunks_added": len(chunks), "source": os.path.basename(file_path)}


def answer_question(question: str, vectordb: VectorStore, top_k=6):
    q_embed = embed_texts([question])[0]

    retrieved = vectordb.hybrid_search(q_embed, question, top_k=top_k)

    context = "\n\n".join(
        [f"[Source: {r['source']}] {r['text']}" for r in retrieved]
    )

    answer = generate_answer_with_fallback(question, context)


    # Convert lines into clean bullet list
    lines = [line.strip("- ").strip() for line in answer.split("\n") if line.strip()]

    return {
    "question": question,
    "answer_raw": answer,   # ✅ ADD THIS
    "summary": lines[0] if lines else answer,
    "points": lines[1:] if len(lines) > 1 else [],
    "sources": list({r["source"] for r in retrieved})
}

from question_parser import split_questions


def answer_multiple_questions(user_query: str, vectordb: VectorStore, top_k=6):
    questions = split_questions(user_query)

    results = []
    for q in questions:
        ans = answer_question(q, vectordb, top_k=top_k)
        results.append(ans)

    return {
        "original_query": user_query,
        "total_questions": len(questions),
        "answers": results
    }
