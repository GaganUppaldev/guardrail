"""main.py - the FastAPI app (front door of the backend)."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.chat import answer

app = FastAPI(title="Guardrail GP Assistant")


class ChatRequest(BaseModel):
    # Pydantic checks the input for us: empty or huge messages are rejected automatically.
    message: str = Field(min_length=1, max_length=1000)
    language: str = "en"


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    flag: str  # "answered" | "no_answer"  (later: "emergency", "medical", "blocked")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return answer(req.message)
