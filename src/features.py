"""Ingeniería de características para el modelado (fase final).

Parte de las 21 características del EDA (`eda_utils`) y agrega:
  * transformaciones decididas en el EDA (log de tamaño, diversidad ajustada,
    banderas de resumen muy corto / muy largo, descarte de variables casi nulas),
  * legibilidad (implementación propia, sin descargas),
  * errores de ortografía,
  * similitud semántica y cobertura de ideas del texto fuente (TF-IDF + LSA).

Nada aquí usa `prompt_id` como variable: el modelo debe generalizar a textos nuevos.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker

import eda_utils as eu

# ---------------------------------------------------------------- constantes
# Variables que casi siempre valen cero (EDA, sección 6.4): se descartan.
DROPPED = ["digit_ratio", "exclamation_count", "question_count"]
# Constantes por texto fuente: actúan como identificador del prompt (con 4 textos, un modelo
# lineal las usa para "reconocer" el texto y extrapola mal al texto nuevo; ver informe).
PROMPT_LEVEL = ["prompt_word_count", "summary_source_word_ratio"]
# Límites IQR del EDA para las banderas.
VERY_SHORT_MAX_WORDS = 28
VERY_LONG_MIN_WORDS = 172

SIZE_VARS = [
    "char_count", "word_count", "sentence_count", "unique_word_count",
    "comma_count",
]

READABILITY = ["flesch_reading_ease", "fk_grade", "gunning_fog", "ari", "syll_per_word",
               "complex_word_ratio"]
SPELLING = ["misspelled_count", "misspelled_ratio"]
SEMANTIC = ["tfidf_cos_source", "lsa_cos_source", "tfidf_cos_question",
            "cov_max_sim", "cov_mean_sim", "cov_frac_sentences", "cov_top3_mean"]

_VOWELS = re.compile(r"[aeiouy]+")
_SPELL = SpellChecker(language="en")


# --------------------------------------------------------------- legibilidad
def count_syllables(word: str) -> int:
    """Conteo heurístico de sílabas (grupos de vocales, sin 'e' muda final)."""
    w = word.lower()
    if not w:
        return 0
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def readability_features(text: str) -> dict:
    words = eu.tokenize_words(text)
    n_w = max(len(words), 1)
    n_s = max(len(eu.split_sentences(text)), 1)
    syl = [count_syllables(w) for w in words] or [1]
    n_syl = sum(syl)
    complex_w = sum(1 for s in syl if s >= 3)
    chars = sum(len(w) for w in words)
    wps, spw = n_w / n_s, n_syl / n_w
    return {
        "flesch_reading_ease": 206.835 - 1.015 * wps - 84.6 * spw,
        "fk_grade": 0.39 * wps + 11.8 * spw - 15.59,
        "gunning_fog": 0.4 * (wps + 100 * complex_w / n_w),
        "ari": 4.71 * (chars / n_w) + 0.5 * wps - 21.43,
        "syll_per_word": spw,
        "complex_word_ratio": complex_w / n_w,
    }


def spelling_features(text: str) -> dict:
    words = [w for w in eu.tokenize_words(text) if len(w) > 2 and "'" not in w]
    if not words:
        return {"misspelled_count": 0, "misspelled_ratio": 0.0}
    bad = _SPELL.unknown(words)
    n_bad = sum(1 for w in words if w in bad)
    return {"misspelled_count": n_bad, "misspelled_ratio": n_bad / len(words)}


# ------------------------------------------------- semántica (TF-IDF y LSA)
@dataclass
class SemanticModel:
    """TF-IDF + LSA ajustados una sola vez (no supervisado, no usa las notas)."""
    n_components: int = 100
    tfidf: TfidfVectorizer = field(default=None, repr=False)
    svd: TruncatedSVD = field(default=None, repr=False)

    def fit(self, texts: list[str]) -> "SemanticModel":
        self.tfidf = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            min_df=2, sublinear_tf=True, token_pattern=r"[A-Za-z']{2,}",
        )
        X = self.tfidf.fit_transform(texts)
        k = min(self.n_components, X.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self

    def _vec(self, texts):
        return self.tfidf.transform(texts)

    @staticmethod
    def _cos(a, b) -> np.ndarray:
        """Coseno fila a fila entre dos matrices (dispersas o densas)."""
        num = np.asarray(a.multiply(b).sum(axis=1)).ravel() if hasattr(a, "multiply") \
            else (a * b).sum(axis=1)
        da = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel()) if hasattr(a, "multiply") \
            else np.linalg.norm(a, axis=1)
        db = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel()) if hasattr(b, "multiply") \
            else np.linalg.norm(b, axis=1)
        return num / np.maximum(da * db, 1e-9)

    def features(self, summaries: list[str], sources: list[str], questions: list[str]) -> pd.DataFrame:
        S, P, Q = self._vec(summaries), self._vec(sources), self._vec(questions)
        out = {
            "tfidf_cos_source": self._cos(S, P),
            "tfidf_cos_question": self._cos(S, Q),
            "lsa_cos_source": self._cos(self.svd.transform(S), self.svd.transform(P)),
        }
        # Cobertura de ideas: similitud del resumen con cada oración del texto fuente.
        cache: dict[str, tuple] = {}
        cols = {k: [] for k in ["cov_max_sim", "cov_mean_sim", "cov_frac_sentences", "cov_top3_mean"]}
        for i, (summ, src) in enumerate(zip(summaries, sources)):
            if src not in cache:
                sents = [s for s in eu.split_sentences(eu.basic_clean(src)) if len(s.split()) >= 4]
                cache[src] = (sents, self._vec(sents) if sents else None)
            sents, M = cache[src]
            if M is None:
                for k in cols:
                    cols[k].append(0.0)
                continue
            sims = self._cos_one_to_many(S[i], M)
            top3 = np.sort(sims)[::-1][:3]
            cols["cov_max_sim"].append(float(sims.max()))
            cols["cov_mean_sim"].append(float(sims.mean()))
            cols["cov_frac_sentences"].append(float((sims > 0.10).mean()))
            cols["cov_top3_mean"].append(float(top3.mean()))
        out.update(cols)
        return pd.DataFrame(out)

    @staticmethod
    def _cos_one_to_many(v, M) -> np.ndarray:
        num = (M @ v.T).toarray().ravel()
        dv = np.sqrt(v.multiply(v).sum())
        dm = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
        return num / np.maximum(dv * dm, 1e-9)


# --------------------------------------------------------------- ensamblado
def build_raw_features(df: pd.DataFrame, prompts: pd.DataFrame, sem: SemanticModel) -> pd.DataFrame:
    """df necesita `text` y `prompt_id`. Devuelve df + todas las características crudas."""
    df = df.copy()
    df["text"] = df["text"].map(eu.basic_clean)
    base = eu.build_feature_frame(df, prompts)
    read = pd.DataFrame(base["text"].map(readability_features).tolist(), index=base.index)
    spell = pd.DataFrame(base["text"].map(spelling_features).tolist(), index=base.index)
    sem_f = sem.features(base["text"].tolist(), base["prompt_text"].tolist(),
                         base["prompt_question"].fillna("").tolist())
    sem_f.index = base.index
    return pd.concat([base, read, spell, sem_f], axis=1)


def to_model_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones del EDA: log de tamaño, diversidad ajustada, banderas, descartes."""
    X = pd.DataFrame(index=raw.index)
    keep = [c for c in eu.NUMERIC_FEATURES if c not in DROPPED + PROMPT_LEVEL]
    for c in keep + READABILITY + SPELLING + SEMANTIC:
        X[c] = raw[c].astype(float)
    for c in SIZE_VARS + ["misspelled_count"]:
        X[f"log_{c}"] = np.log1p(X.pop(c))
    # Diversidad léxica ajustada por tamaño: índice de Guiraud (únicas / raíz de palabras)
    X["guiraud"] = raw["unique_word_count"] / np.sqrt(raw["word_count"].clip(lower=1))
    X = X.drop(columns=["type_token_ratio"])
    X["flag_very_short"] = (raw["word_count"] <= VERY_SHORT_MAX_WORDS).astype(int)
    X["flag_very_long"] = (raw["word_count"] >= VERY_LONG_MIN_WORDS).astype(int)
    return X


# Subconjuntos por modelo
def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    base = [c for c in columns if c not in
            set(READABILITY + SPELLING + SEMANTIC + ["log_misspelled_count"])]
    return {"base21": base, "enriquecido": list(columns)}


def length_only(columns: list[str]) -> list[str]:
    return ["log_word_count"]
