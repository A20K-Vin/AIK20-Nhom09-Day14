from typing import List, Dict

from llm.client import LLMClient
from llm.prompt import build_rag_messages

class Generator:
    def __init__(self, client: LLMClient):
        self.client = client

    async def generate(self, question: str, contexts: List[Dict]) -> Dict:
        context_texts = [c["text"] for c in contexts]
        messages = build_rag_messages(question, context_texts)
        answer = await self.client.complete(messages)

        sources = list({c.get("source", "") for c in contexts if c.get("source")})
        return {
            "answer": answer,
            "contexts": context_texts,
            "metadata": {
                "model": self.client.model,
                "sources": sources,
            },
        }
