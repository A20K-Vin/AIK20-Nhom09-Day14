import os
from typing import List
from openai import OpenAI


class Embedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dim = 1536

    def embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            results.extend(item.embedding for item in response.data)
        return results

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
