"""chat.py - turns a question into an answer using RAG (search docs -> ask Llama)."""
from app.llm.client import ask_llm
from app.rag.search import search

MIN_SCORE = 0.35  # below this, the docs don't cover the question -> "I don't know"

I_DONT_KNOW = (
    "Sorry, I don't have information about that. "
    "Please call reception on 0116 496 0123."
)

RULES = """You are the admin assistant for Riverside Lane Surgery.
Rules:
- Answer ONLY using the DOCUMENTS below. If they don't contain the answer, say you don't know.
- Never invent times, numbers, phone numbers or policies.
- Never give medical advice or a diagnosis.
- Keep answers short, friendly and clear (2-4 sentences)."""


def answer(message: str) -> dict:
    hits = search(message)

    # Guardrail: nothing relevant found -> don't let the LLM guess.
    if not hits or hits[0]["score"] < MIN_SCORE:
        return {"reply": I_DONT_KNOW, "sources": [], "flag": "no_answer"}

    relevant = [h for h in hits if h["score"] >= MIN_SCORE]
    documents = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in relevant)
    prompt = f"{RULES}\n\nDOCUMENTS:\n{documents}\n\nQUESTION: {message}\n\nANSWER:"

    reply = ask_llm(prompt)
    sources = list(dict.fromkeys(h["source"] for h in relevant))  # unique, keep order
    return {"reply": reply.strip(), "sources": sources, "flag": "answered"}
