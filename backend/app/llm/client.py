
import ollama

MODEL = "llama3.1:8b"


def ask_llm(prompt: str) -> str:
    """
    Sends a prompt to the Ollama language model
    and returns the model's text response.
    """

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0}
    )

    return response["message"]["content"]


