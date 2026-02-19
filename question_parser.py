import re

def split_questions(user_query: str):
    """
    Split multi-question text into individual questions.
    Handles:
    - ?, newline
    - "and", "also"
    - numbered lists
    """
    text = user_query.strip()

    # Replace newlines with space
    text = text.replace("\n", " ")

    # Split by question marks first
    parts = re.split(r"\?\s*", text)

    questions = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        # Add '?' back if it looked like a question
        if not p.endswith("?"):
            p = p + "?"

        questions.append(p)

    # If still only 1, try splitting by "also" / "and"
    if len(questions) == 1:
        parts = re.split(r"\s+(?:also|and)\s+", text, flags=re.IGNORECASE)
        questions = [p.strip() for p in parts if p.strip()]

        # Add '?' if missing
        questions = [q if q.endswith("?") else q + "?" for q in questions]

    return questions
