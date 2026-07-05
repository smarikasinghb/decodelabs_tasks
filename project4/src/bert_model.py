from functools import lru_cache
from typing import Tuple

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except ImportError as error:
    raise ImportError(
        "The transformers package is required for BERT inference. "
        "Install it with `pip install transformers torch`."
    ) from error

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"


def _load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)


@lru_cache(maxsize=1)
def _get_pipeline():
    return _load_pipeline()


def predict_bert(text: str) -> Tuple[str, float]:
    """Predict sentiment with a pretrained BERT model and return label + confidence."""
    if not text or not isinstance(text, str):
        return "Neutral", 0.0

    classifier = _get_pipeline()
    result = classifier(text[:512], truncation=True)
    if not result:
        return "Neutral", 0.0

    output = result[0]
    label_text = str(output.get("label", "3 stars")).strip()
    score = float(output.get("score", 0.0))

    if label_text.startswith("1") or label_text.startswith("2"):
        sentiment = "Negative"
    elif label_text.startswith("3"):
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    return sentiment, score
