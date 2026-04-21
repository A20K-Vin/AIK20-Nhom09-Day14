from typing import Dict, List

SYSTEM_PROMPT = (
    "Bạn là trợ lý hỗ trợ khách hàng chuyên nghiệp. "
    "Hãy trả lời câu hỏi DỰA TRÊN ngữ cảnh được cung cấp. "
    "Nếu ngữ cảnh không đủ thông tin, hãy trả lời: "
    "\"Tôi không có đủ thông tin để trả lời câu hỏi này.\" "
    "Không được bịa đặt thông tin."
)

USER_TEMPLATE = """\
Ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời:"""


def build_rag_messages(question: str, contexts: List[Dict]) -> List[dict]:
    context_str = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context_str, question=question)},
    ]
