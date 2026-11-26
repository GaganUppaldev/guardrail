from fastapi import FastAPI

app = FastAPI(title="Gaurdrail GP Assistent")

@app.get("/health")
def health():
    """check server is alive"""
    return {"status" : "ok" }