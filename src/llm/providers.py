import asyncio
from src.llm.clients import (
    openai_client, deepseek_client, gemini_client,
    SEM_OPENAI, SEM_DEEPSEEK, SEM_GEMINI
)
from src.rag.prompt import MODEL_CONTEXT
from src.config import setup_logger

logger = setup_logger(__name__)

async def ask_openai(model: str, prompt: str) -> str:
    async with SEM_OPENAI:
        try:
            resp = await openai_client.responses.create(
                model=model,
                input=prompt,
            )
            return resp.output_text.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return ""

async def ask_deepseek(model: str, prompt: str) -> str:
    async with SEM_DEEPSEEK:
        try:
            resp = await asyncio.to_thread(
                lambda: deepseek_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": MODEL_CONTEXT},
                        {"role": "user", "content": prompt},
                    ],
                )
            )
            return resp.choices[0].message.content.strip() if resp.choices[0].message.content else " "
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            return ""

async def ask_gemini(model: str, prompt: str) -> str:
    async with SEM_GEMINI:
        try:
            resp = await gemini_client.aio.models.generate_content(
                model=model,
                contents=prompt,
            )
            return resp.text.strip() if resp.text else " "
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return ""
