from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import CHROMA_PATH, CHROMA_COLLECTION, EMBED_MODEL

def test_chroma():
    print("Loading DB...")

    emb = OpenAIEmbeddings(model=EMBED_MODEL)

    db = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH,
        embedding_function=emb
    )

    print("Fetching stored chunks...")
    data = db.get(include=["documents", "embeddings", "metadatas"])

    total = len(data["ids"])
    print(f"Total IDs: {total}")
    print(f"Total docs: {len(data['documents'])}")
    print(f"Total embeddings: {len(data['embeddings'])}")

    if total == 0:
        print("ERROR: No embeddings were found.")
        return

    print("\nTesting retrieval...")
    query = "Como fazer manejo de pragas em maçãs?"
    docs = db.similarity_search(query, k=5)

    for i, d in enumerate(docs):
        print(f"\n--- Result {i+1} ---")
        print(d.page_content[:400], "...")
        print("\nMetadata:", d.metadata)

if __name__ == "__main__":
    test_chroma()
