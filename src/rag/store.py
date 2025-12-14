import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import CHROMA_PATH, CHROMA_COLLECTION, setup_logger

logger = setup_logger(__name__)

def load_retriever():
    logger.info(f"Loading vector store from {CHROMA_PATH}...")
    start = time.perf_counter()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    logger.info(f"Vector store loaded in {time.perf_counter() - start:.2f}s.")
    return retriever
