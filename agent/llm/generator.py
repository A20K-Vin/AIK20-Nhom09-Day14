from typing import List, Dict

from llm.client import LLMClient
from llm.prompt import build_rag_messages

class Generator:
    def __init__(self, client: LLMClient):
        self.client = client

    async def generate(self, question: str, contexts: List[Dict]) -> Dict:
        # Giữ nguyên tất cả các trường context để in chunk_id, source, section, score...
        context_items = [
            {
                "text": c["text"],
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "section": c.get("section"),
                "score": c.get("score")
            }
            for c in contexts
        ]
        messages = build_rag_messages(question, [c["text"] for c in context_items])
        answer = await self.client.complete(messages)

        sources = list({c.get("source", "") for c in context_items if c.get("source")})
        return {
            "answer": answer,
            "contexts": context_items,
            "metadata": {
                "model": self.client.model,
                "sources": sources,
            },
        }
