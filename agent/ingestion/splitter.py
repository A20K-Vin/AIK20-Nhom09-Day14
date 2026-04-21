from typing import List, Dict
import tiktoken


class TokenTextSplitter:
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        model_name: str = "gpt-4.1-mini",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def split(self, docs: List[Dict]) -> List[Dict]:
        chunks = []

        for doc in docs:
            token_chunks = self._split_text(doc["text"])

            for chunk_text in token_chunks:
                chunks.append({
                    **doc,
                    "text": chunk_text
                })

        return chunks

    def _split_text(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)

        chunks = []
        start = 0
        n_tokens = len(tokens)

        while start < n_tokens:
            end = min(start + self.chunk_size, n_tokens)
            chunk_tokens = tokens[start:end]

            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += self.chunk_size - self.chunk_overlap

        return chunks
