from collections.abc import Mapping

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def health() -> Mapping[str, str]:
    return {"status": "ok"}
