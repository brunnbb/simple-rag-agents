import os
import logging
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# Load environment variables from .env file
load_dotenv(override=True)

# Embedding model
EMBED_MODEL = "text-embedding-3-small"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths 
CHROMA_PATH = "chroma_db"
CHROMA_COLLECTION = "chunks"
JSONL_PATH = r"data/chunks_out.jsonl"
PERGUNTAS_PATH = r"data/perguntas_e_respostas_rag.csv"

LOG_PATH = "logs"
LOG_FILE = os.path.join(LOG_PATH, "pipeline.log")

os.makedirs(LOG_PATH, exist_ok=True)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(name: str = "rag_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger 

    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    for lib in ("httpx", "openai", "google", "urllib3"):
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.INFO)
        lib_logger.propagate = False
        
        if not lib_logger.handlers:
            lib_logger.addHandler(handler)

    return logger
