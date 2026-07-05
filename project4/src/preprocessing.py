import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """Normalize a sentence for text-based sentiment modeling."""
    if text is None:
        return ""

    text_str = str(text).lower()
    text_str = re.sub(r'https?://\S+|www\.\S+', ' ', text_str)
    text_str = re.sub(r'@[A-Za-z0-9_]+', ' ', text_str)
    text_str = re.sub(r'#[A-Za-z0-9_]+', ' ', text_str)
    text_str = re.sub(r'[^a-z0-9\s]', ' ', text_str)
    text_str = re.sub(r'([a-z0-9])\1{2,}', r'\1\1', text_str)
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    return text_str
