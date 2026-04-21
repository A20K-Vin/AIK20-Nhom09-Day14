from typing import List, Dict
import re

try:
    import tiktoken
except ImportError:
    tiktoken = None


class SectionTokenTextSplitter:
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        model_name: str = "gpt-4.1-mini",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if tiktoken is None:
            self.encoding = None
        else:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")

    def split(self, docs: List[Dict]) -> List[Dict]:
        chunks = []

        for doc in docs:
            sections = self._split_sections(doc["text"])

            for section_title, section_text in sections:
                sub_chunks = self._split_if_needed(section_text)

                for chunk_text in sub_chunks:
                    chunks.append({
                        **doc,
                        "text": chunk_text,
                        "section": section_title,
                        "id": section_title
                    })

        return chunks

    # =========================
    # 1. Split theo section
    # =========================
    def _split_sections(self, text: str):
        pattern = r"(#{2,4} .+)"  # match ##, ###, ####

        parts = re.split(pattern, text)

        sections = []
        current_title = "intro"
        current_text = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(pattern, part):
                if current_text:
                    sections.append((current_title, current_text.strip()))
                current_title = part
                current_text = ""
            else:
                current_text += "\n" + part

        if current_text:
            sections.append((current_title, current_text.strip()))

        return sections

    # =========================
    # 2. Token split nếu quá dài
    # =========================
    def _split_if_needed(self, text: str) -> List[str]:
        if self.encoding is None:
            words = text.split()
            if len(words) <= self.chunk_size:
                return [text]

            chunks = []
            start = 0
            step = max(1, self.chunk_size - self.chunk_overlap)
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunks.append(" ".join(words[start:end]))
                start += step
            return chunks

        tokens = self.encoding.encode(text)

        if len(tokens) <= self.chunk_size:
            return [text]

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
