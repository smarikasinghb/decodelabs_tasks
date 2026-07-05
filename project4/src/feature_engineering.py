import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from typing import Tuple

from .emoji_features import extract_emoji_sentiment


def build_text_vectorizer(texts, max_features: int = 2000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vectorizer.fit(texts)
    return vectorizer


def build_emoji_matrix(texts):
    features = [extract_emoji_sentiment(text) for text in texts]
    matrix = np.array(
        [
            [row["emoji_positive"], row["emoji_negative"], row["emoji_total"], row["emoji_score"]]
            for row in features
        ],
        dtype=float,
    )
    return matrix


def build_features(df, text_column: str = "clean_sentence") -> Tuple[sparse.spmatrix, TfidfVectorizer]:
    if text_column not in df.columns:
        raise ValueError(f"Missing feature column: {text_column}")

    texts = df[text_column].astype(str).tolist()
    vectorizer = build_text_vectorizer(texts)
    text_matrix = vectorizer.transform(texts)
    emoji_matrix = build_emoji_matrix(df["sentence"].astype(str).tolist())
    emoji_sparse = sparse.csr_matrix(emoji_matrix)
    combined = sparse.hstack([text_matrix, emoji_sparse], format="csr")
    return combined, vectorizer


def transform_features(df, vectorizer: TfidfVectorizer, text_column: str = "clean_sentence") -> sparse.spmatrix:
    if text_column not in df.columns:
        raise ValueError(f"Missing feature column: {text_column}")

    try:
        text_matrix = vectorizer.transform(df[text_column].astype(str).tolist())
    except NotFittedError as error:
        raise ValueError("The provided vectorizer is not fitted.") from error

    emoji_matrix = build_emoji_matrix(df["sentence"].astype(str).tolist())
    emoji_sparse = sparse.csr_matrix(emoji_matrix)
    combined = sparse.hstack([text_matrix, emoji_sparse], format="csr")
    return combined
