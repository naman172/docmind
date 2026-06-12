import asyncio
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from ragas import (
    EvaluationDataset,
    MultiTurnSample,
    RunConfig,
    SingleTurnSample,
    evaluate,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from docmind_evals.constants import RAGAS_COLLECTION_NAME

load_dotenv()

local_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

ragas_llm = LangchainLLMWrapper(local_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

faithfulness_metric = Faithfulness(llm=ragas_llm)

metrics = [faithfulness_metric]

config = RunConfig(timeout=180, max_workers=1)


async def main() -> None:
    data_path = Path(__file__).parent.parent.parent / "data" / "fastapi_eval.json"
    questions = json.loads(data_path.read_text())

    dataset: list[SingleTurnSample | MultiTurnSample] = []

    async with httpx.AsyncClient(
        base_url="http://localhost:8000", timeout=60.0
    ) as client:
        for question in questions:
            response = await client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": question["question"]}],
                    "collection_name": RAGAS_COLLECTION_NAME,
                    "stream": False,
                },
            )

            print(f"✓ {question['question'][:60]}")

            dataset.append(
                SingleTurnSample(
                    user_input=question["question"],
                    response=response.json()["response"],
                    retrieved_contexts=[
                        chunk["text"] for chunk in response.json()["retrieved_chunks"]
                    ],
                    reference=question["grounding_truth"],
                )
            )

    print(f"\nDataset built. Running RAGAS eval on {len(dataset)} samples...")
    result = evaluate(
        dataset=EvaluationDataset(dataset), metrics=metrics, run_config=config
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
