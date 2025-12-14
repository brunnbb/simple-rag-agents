import asyncio
import time

from src.pipeline.csv_processor import process_csv
from src.config import PERGUNTAS_PATH
from src.llm.clients import openai_client, gemini_client

async def main():
    start = time.perf_counter()
    try:
        await process_csv(PERGUNTAS_PATH)
    finally:
        await asyncio.gather(
            openai_client.close(),
            gemini_client.aio.aclose(),
            return_exceptions=True
        )
    print(f"Tempo total: {time.perf_counter() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
