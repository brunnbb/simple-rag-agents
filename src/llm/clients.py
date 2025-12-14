import asyncio
from openai import AsyncOpenAI, OpenAI
from google import genai
from src.config import OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SEM_OPENAI = asyncio.Semaphore(6)
SEM_DEEPSEEK = asyncio.Semaphore(4)
SEM_GEMINI = asyncio.Semaphore(4)
