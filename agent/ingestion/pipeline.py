from typing import List

from ingestion.loader import DocumentLoader
from ingestion.splitter import TokenTextSplitter
from ingestion.embedder import Embedder


class IngestionPipeline:
    def __init__(self, vector_store, chunk_size: int = 512, chunk_overlap: int = 64):
        self.loader = DocumentLoader()
        self.splitter = TokenTextSplitter(chunk_size, chunk_overlap)
        self.embedder = Embedder()
        self.vector_store = vector_store

    def ingest(self, paths: List[str]) -> int:
        docs = []
        for path in paths:
            docs.extend(self.loader.load(path))

        chunks = self.splitter.split(docs)
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)

        self.vector_store.add(texts, embeddings, chunks)
        self.vector_store.save()
        return len(chunks)
