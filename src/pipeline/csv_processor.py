import asyncio
import pandas as pd
from tqdm.asyncio import tqdm

from src.rag.prompt import build_rag_prompt
from src.rag.store import load_retriever
from src.llm.providers import ask_openai, ask_gemini, ask_deepseek
from src.llm.retry import retry
from src.config import setup_logger

logger = setup_logger(__name__)

MODELS = {
    "gpt-5": ask_openai,
    "gpt-5-mini": ask_openai,
    "gpt-5-nano": ask_openai,
    "gemini-2.5-flash": ask_gemini,
    "gemini-2.5-pro": ask_gemini,
    "deepseek-chat": ask_deepseek,
    "deepseek-reasoner": ask_deepseek,
}

async def process_csv(filename: str):
    retriever = load_retriever()
    df = pd.read_csv(filename)

    logger.info(f"Processando {len(df)} perguntas")

    for model in MODELS.keys():
        if model not in df.columns:
            df[model] = ""
        df[model] = df[model].fillna("").astype(str)

    sem = asyncio.Semaphore(3)

    async def process_row(i, question):
        async with sem:
            logger.info(f"→ Pergunta {i+1}: processando...")
            rag_prompt = await build_rag_prompt(question, retriever)

            async def query(model, fn):
                if str(df.at[i, model]).strip():
                    return df.at[i, model]
                return await retry(fn, model, rag_prompt)

            results = await asyncio.gather(
                *[query(m, fn) for m, fn in MODELS.items()]
            )

            for model, result in zip(MODELS.keys(), results):
                df.at[i, model] = result

            df.to_csv(filename, index=False, encoding="utf-8")
            logger.info(f"✓ Pergunta {i+1}: salva.")

    tasks = [
        process_row(i, row["pergunta"].strip())
        for i, row in df.iterrows()
        if str(row.get("pergunta", "")).strip()
    ]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await coro
