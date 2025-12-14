import asyncio
from typing import Callable
from src.config import setup_logger

logger = setup_logger(__name__)

async def retry(fn: Callable, *args, retries=3, delay=10, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                logger.exception(f"Error after retries: {e}")
                return "ERROR"
            sleep_time = delay * attempt
            logger.warning(f"Retry {attempt}/{retries} in {sleep_time}s")
            await asyncio.sleep(sleep_time)
