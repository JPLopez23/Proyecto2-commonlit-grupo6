"""Limpieza e ingeniería de características de texto para el EDA de CommonLit."""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np
import pandas as pd

# Stopwords en inglés (lista mínima para no depender de descargas de nltk)
ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was",
    "were", "be", "been", "being", "of", "to", "in", "on", "for", "with", "as",
    "at", "by", "from", "that", "this", "these", "those", "it", "its", "he",
    "she", "they", "them", "his", "her", "their", "we", "you", "i", "not", "no",
    "so", "than", "then", "there", "here", "which", "who", "whom", "what",
    "when", "where", "why", "how", "all", "any", "both", "each", "more", "most",
    "other", "some", "such", "only", "own", "same", "too", "very", "can",
    "will", "just", "do", "does", "did", "has", "have", "had", "having", "up",
    "down", "out", "about", "into", "over", "after", "before", "again",
}

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_SPLIT_RE = re.compile(r"[.!?]+")


def basic_clean(text: str) -> str:
    """Homogeneiza espacios y quita caracteres de control; conserva puntuación y mayúsculas."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = "".join(ch for ch in text if ch == "\t" or ch >= " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]


def summary_features(text: str) -> dict:
    clean = basic_clean(text)
    words = tokenize_words(clean)
    n_words = len(words)
    n_chars = len(clean)
    n_sent = max(len(split_sentences(clean)), 1)
    unique = set(words)

    word_lengths = [len(w) for w in words] or [0]
    stop_count = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    punct_count = sum(1 for ch in clean if ch in string.punctuation)
    upper_count = sum(1 for ch in clean if ch.isupper())
    digit_count = sum(1 for ch in clean if ch.isdigit())

    return {
        "char_count": n_chars,
        "word_count": n_words,
        "sentence_count": n_sent,
        "unique_word_count": len(unique),
        "type_token_ratio": len(unique) / n_words if n_words else 0.0,
        "avg_word_length": float(np.mean(word_lengths)),
        "avg_sentence_length_words": n_words / n_sent,
        "stopword_ratio": stop_count / n_words if n_words else 0.0,
        "punct_per_word": punct_count / n_words if n_words else 0.0,
        "uppercase_ratio": upper_count / n_chars if n_chars else 0.0,
        "digit_ratio": digit_count / n_chars if n_chars else 0.0,
        "exclamation_count": clean.count("!"),
        "question_count": clean.count("?"),
        "comma_count": clean.count(","),
    }


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def overlap_features(summary: str, prompt_text: str) -> dict:
    """Solapamiento léxico resumen vs. texto fuente (parafraseo vs. copiar y pegar)."""
    s_tokens = [w for w in tokenize_words(summary) if w not in ENGLISH_STOPWORDS]
    p_tokens = [w for w in tokenize_words(prompt_text) if w not in ENGLISH_STOPWORDS]
    s_set, p_set = set(s_tokens), set(p_tokens)

    if not s_set:
        return {
            "jaccard_unigram": 0.0,
            "prop_words_in_source": 0.0,
            "bigram_overlap_ratio": 0.0,
            "trigram_overlap_ratio": 0.0,
            "new_word_ratio": 0.0,
        }

    inter = s_set & p_set
    union = s_set | p_set
    s_bi, p_bi = _ngrams(s_tokens, 2), _ngrams(p_tokens, 2)
    s_tri, p_tri = _ngrams(s_tokens, 3), _ngrams(p_tokens, 3)

    return {
        "jaccard_unigram": len(inter) / len(union),
        "prop_words_in_source": len(inter) / len(s_set),
        "bigram_overlap_ratio": sum((s_bi & p_bi).values()) / max(sum(s_bi.values()), 1),
        "trigram_overlap_ratio": sum((s_tri & p_tri).values()) / max(sum(s_tri.values()), 1),
        "new_word_ratio": 1 - len(inter) / len(s_set),
    }


def build_feature_frame(summaries: pd.DataFrame,
                        prompts: pd.DataFrame,
                        text_col: str = "text",
                        prompt_text_col: str = "prompt_text") -> pd.DataFrame:
    """Une resúmenes con prompts por prompt_id y agrega todas las características."""
    df = summaries.merge(prompts, on="prompt_id", how="left")
    feat_rows = df[text_col].apply(summary_features).apply(pd.Series)
    ov_rows = df.apply(
        lambda r: overlap_features(r[text_col], r.get(prompt_text_col, "")), axis=1
    ).apply(pd.Series)
    out = pd.concat([df, feat_rows, ov_rows], axis=1)

    out["prompt_word_count"] = out[prompt_text_col].fillna("").map(
        lambda t: len(tokenize_words(t))
    )
    out["summary_source_word_ratio"] = (
        out["word_count"] / out["prompt_word_count"].replace(0, np.nan)
    )
    return out


NUMERIC_FEATURES = [
    "char_count", "word_count", "sentence_count", "unique_word_count",
    "type_token_ratio", "avg_word_length", "avg_sentence_length_words",
    "stopword_ratio", "punct_per_word", "uppercase_ratio", "digit_ratio",
    "exclamation_count", "question_count", "comma_count",
    "jaccard_unigram", "prop_words_in_source", "bigram_overlap_ratio",
    "trigram_overlap_ratio", "new_word_ratio",
    "prompt_word_count", "summary_source_word_ratio",
]

TARGETS = ["content", "wording"]
