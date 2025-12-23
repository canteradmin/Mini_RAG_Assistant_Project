
MAX_HISTORY_TURNS = 4

SYSTEM_PROMPT = """
You are a retrieval-augmented assistant.
Answer ONLY using the provided context.
If the answer is not present, say "I don't have enough information to answer that question"
"""

def build_messages(question: str, context: str, history: list) -> list:
    """
    Build conversation messages with history and RAG context.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent conversation history
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["query"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    # Current query with context
    messages.append({
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{question}

Answer:
"""
    })
    return messages