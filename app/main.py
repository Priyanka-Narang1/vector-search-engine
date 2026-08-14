from fastapi import FastAPI

app = FastAPI(title="Distributed Vector Search Engine")


@app.get("/health")
def health_check():
    return {"status": "ok"}
