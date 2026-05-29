import logging
import os
import time
from typing import cast

import httpx
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
logger = logging.getLogger(__name__)


async def embed_texts(
    texts: list[str], model: str = "nomic-embed-text"
) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            start_time = time.perf_counter()

            response = await client.post(
                OLLAMA_BASE_URL + "/api/embed", json={"model": model, "input": texts}
            )
            response.raise_for_status()

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            logger.info("embedded %d texts in %.1fms", len(texts), latency_ms)
            return cast(list[list[float]], response.json()["embeddings"])

        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")
