import json
import os
import sys
import time
from typing import Iterable, Any

from tqdm import tqdm
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings

from src.config import (
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    JSONL_PATH
)


def count_lines(path: str) -> int:
    """Count total lines in JSONL for tqdm progress bar."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def load_jsonl_stream(path: str) -> Iterable[dict[str, Any]]:
    """Stream JSONL file line-by-line."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    except Exception as e:
        print(f"[ingest] Error reading JSONL file: {e}")
        raise

def ensure_db(mode: str, embeddings: OpenAIEmbeddings) -> Chroma:
    """Prepare Chroma according to the mode."""
    os.makedirs(CHROMA_PATH, exist_ok=True)

    if mode == "create":
        print("[ingest] Rebuilding vector DB from scratch...")
        Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        ).delete_collection()

        return Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
        )

    elif mode == "append":
        print("[ingest] Appending to existing vector DB...")
        return Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
        )

    else:
        raise ValueError("mode must be 'create' or 'append'")

def ingest(mode: str = "create", batch_size: int = 128) -> None:
    start_time = time.time()
    print(f"[ingest] Mode: {mode}")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = ensure_db(mode, embeddings)

    # Load existing ids for append mode
    existing_ids = set()
    if mode == "append":
        print("[ingest] Loading existing chunk_ids for dedup...")
        existing_docs = vectorstore.get(include=["metadatas", "documents", "embeddings"])
        if existing_docs and "ids" in existing_docs:
            existing_ids = set(existing_docs["ids"])
        print(f"[ingest] Existing chunks: {len(existing_ids)}")

    total_lines = count_lines(JSONL_PATH)
    print(f"[ingest] Total chunks in file: {total_lines}")

    stream = load_jsonl_stream(JSONL_PATH)

    buffer_texts: list[str] = []
    buffer_ids: list[str] = []
    buffer_meta: list[dict[str, Any]] = []

    total_added = 0
    total_skipped = 0

    print("[ingest] Starting ingestion...")

    progress = tqdm(stream, total=total_lines, desc="Embedding", unit="chunk")

    for chunk in progress:
        cid = chunk["chunk_id"]

        if cid in existing_ids:
            total_skipped += 1
            continue

        buffer_texts.append(chunk["texto"])
        buffer_ids.append(cid)
        buffer_meta.append({
            "doc_id": chunk["doc_id"],
            "titulo": chunk["titulo"],
            "pagina": chunk["pagina"],
            "fonte": chunk["fonte"],
            "path_pdf": chunk["path_pdf"],
        })

        if len(buffer_texts) >= batch_size:
            batch_start = time.time()

            vectorstore.add_texts(
                texts=buffer_texts,
                ids=buffer_ids,
                metadatas=buffer_meta
            )

            batch_time = time.time() - batch_start
            total_added += len(buffer_texts)

            buffer_texts.clear()
            buffer_ids.clear()
            buffer_meta.clear()

            progress.set_postfix({
                "added": total_added,
                "skipped": total_skipped,
                "batch_sec": f"{batch_time:.2f}"
            })

    # Tail
    if buffer_texts:
        vectorstore.add_texts(
            texts=buffer_texts,
            ids=buffer_ids,
            metadatas=buffer_meta
        )
        total_added += len(buffer_texts)

    total_time = time.time() - start_time

    print("\n[ingest] ✔ Completed.")
    print(f"[ingest] Added new chunks: {total_added}")
    print(f"[ingest] Skipped: {total_skipped}")
    print(f"[ingest] Total time: {total_time:.2f} seconds ({total_time/60:.2f} min)")

if __name__ == "__main__":
    mode = "create"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    ingest(mode=mode)
