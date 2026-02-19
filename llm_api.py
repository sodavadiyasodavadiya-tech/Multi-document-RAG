import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"

def generate_answer(question: str, context: str) -> str:
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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()
