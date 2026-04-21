import asyncio
from typing import Dict, List, Optional

from dotenv import load_dotenv
import os


from .ingestion.embedder import Embedder
from .ingestion.pipeline import IngestionPipeline
from .retrieval.vector_store import VectorStore
from .retrieval.retriever import Retriever
from .retrieval.reranker import Reranker
from .llm.client import LLMClient
from .llm.generator import Generator

load_dotenv()
DOCS_FOLDER_PATH = "data/docs"

class MainAgent:
    def __init__(
        self,
        documents: Optional[List[str]] = None,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
        rerank_top_k: int = 3,
        persist_path: str = ".vector_store",
    ):
        self.name = "SupportAgent-v1"
        self.rerank_top_k = rerank_top_k

        embedder = Embedder()
        self.vector_store = VectorStore(persist_path=persist_path)
        self.retriever = Retriever(self.vector_store, embedder, top_k=top_k)
        self.reranker = Reranker()
        self.generator = Generator(LLMClient(model=model))

        if documents:
            pipeline = IngestionPipeline(self.vector_store)
            n = pipeline.ingest(documents)
            print(f"[MainAgent] Ingested {n} chunks from {len(documents)} document(s).")
        else:
            try:
                self.vector_store.load()
                print(f"[MainAgent] Loaded vector store from '{persist_path}'.")
            except Exception:
                print("[MainAgent] No existing vector store found — provide documents to ingest.")

    async def query(self, question: str) -> Dict:
        candidates = self.retriever.retrieve(question)
        reranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)
        return await self.generator.generate(question, reranked)


if __name__ == "__main__":
    docs = []
    for root, _, files in os.walk(DOCS_FOLDER_PATH):
        for file in files:
            if file.lower().endswith((".pdf", ".txt", ".md")):
                docs.append(os.path.join(root, file))
    if not docs:
        print(f"No documents found in '{DOCS_FOLDER_PATH}'. Please add some files to ingest.")

    agent = MainAgent(documents=docs)

    async def test():
        resp = await agent.query("Điều kiện được hoàn tiền là như nào?")
        
        answer = resp["answer"]
        contexts = resp["contexts"]
        metadata = resp["metadata"]

        print("\n=== Question ===")
        print("Điều kiện được hoàn tiền là như nào?")
        print()
        print("\n=== Answer ===")
        print(answer)
        print()
        print("\n=== Retrieved Contexts ===")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n--- Context {i} ---")
            print(ctx)
        print()
        print("\n=== Metadata ===")
        print(metadata)

    asyncio.run(test())
