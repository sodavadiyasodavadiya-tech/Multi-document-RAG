from llm_api import generate_answer  # Gemini
from llm_groq import generate_answer_groq


def generate_answer_with_fallback(question: str, context: str) -> str:
    """
    Try Gemini first, if it fails -> fallback to Groq Llama
    """
    try:
        return generate_answer(question, context)  # ✅ Gemini
    except Exception as e:
        print("⚠️ Gemini failed, switching to Groq fallback...")
        print("Gemini Error:", str(e))

        return generate_answer_groq(question, context)  # ✅ Groq
