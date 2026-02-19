import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Best fast + strong model
MODEL_NAME = "llama-3.1-8b-instant"
# You can also use:
# MODEL_NAME = "llama-3.1-70b-versatile"


def generate_answer_groq(question: str, context: str) -> str:
    prompt = f"""
You are a highly accurate document-based assistant.

RULES:
- Answer ONLY using the provided CONTEXT.
- Do NOT use outside knowledge.
- If answer not found, say: Not found in the document.

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
