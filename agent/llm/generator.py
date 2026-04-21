from typing import List, Dict

from .client import LLMClient
from .prompt import build_rag_messages

class Generator:
    def __init__(self, client: LLMClient):
        self.client = client

    async def generate(self, question: str, contexts: List[Dict]) -> Dict:
        context_texts = [c["text"] for c in contexts]
        messages = build_rag_messages(question, context_texts)
        completion = await self.client.complete(messages)
        answer = completion["content"]

        sources = list({c.get("source", "") for c in contexts if c.get("source")})
        retrieved_ids = []
        for idx, context in enumerate(contexts, start=1):
            context_id = (
                context.get("ground_truth_id")
                or context.get("chunk_id")
                or context.get("id")
                or f"{context.get('source', 'unknown')}#p{context.get('page', 0)}#{idx}"
            )
            retrieved_ids.append(str(context_id))
        return {
            "answer": answer,
            "contexts": context_texts,
            "metadata": {
                "model": self.client.model,
                "sources": sources,
                "retrieved_ids": retrieved_ids,
                "usage": completion["usage"],
            },
        }
