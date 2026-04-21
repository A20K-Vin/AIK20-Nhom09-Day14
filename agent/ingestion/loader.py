from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


class DocumentLoader:
    def load(self, path: str) -> List[Dict]:
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            return self._load_pdf(p)
        if p.suffix.lower() in (".txt", ".md"):
            return self._load_text(p)
        raise ValueError(f"Unsupported file type: {p.suffix}")

    def _load_pdf(self, path: Path) -> List[Dict]:
        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append({"text": text, "source": str(path), "page": i + 1})
        return docs

    def _load_text(self, path: Path) -> List[Dict]:
        text = path.read_text(encoding="utf-8").strip()
        return [{"text": text, "source": str(path), "page": 1}]
