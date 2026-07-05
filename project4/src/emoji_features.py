from typing import Dict, Optional

EMOJI_SENTIMENT = {
    "😀": 1,
    "😃": 1,
    "😄": 1,
    "😁": 1,
    "😊": 1,
    "😍": 1,
    "👍": 1,
    "🎉": 1,
    "😂": 1,
    "😢": -1,
    "😞": -1,
    "😠": -1,
    "😡": -1,
    "👎": -1,
    "💔": -1,
    "😩": -1,
    "😭": -1,
    "😕": -1,
}


def extract_emoji_sentiment(text: Optional[str]) -> Dict[str, int]:
    """Return a small set of emoji sentiment features from raw text."""
    text_str = "" if text is None else str(text)
    positive = 0
    negative = 0

    for character in text_str:
        value = EMOJI_SENTIMENT.get(character)
        if value == 1:
            positive += 1
        elif value == -1:
            negative += 1

    return {
        "emoji_positive": positive,
        "emoji_negative": negative,
        "emoji_total": positive + negative,
        "emoji_score": positive - negative,
    }
