"""Inferencia para la app: recibe un resumen (y su texto fuente) y devuelve las notas
predichas por cada modelo. Todo el preprocesamiento (limpieza, rasgos, TF-IDF/LSA) ocurre aquí,
sin que el usuario lo vea."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import eda_utils as eu
import features as F

ROOT = Path(__file__).resolve().parents[1]
MODELS, RES, DATA = ROOT / "models", ROOT / "results", ROOT / "data"
T = ["content", "wording"]
FEATURE_MODELS = ["baseline_longitud", "ridge_enriquecido", "lgbm_enriquecido"]


class DebertaPredictor:
    """Carga el modelo ajustado por train_deberta.py (carpeta deberta_<tag>_final)."""

    def __init__(self, folder: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        cfg = json.loads((folder / "config.json").read_text())
        self.max_len, self.include_source = cfg["max_len"], cfg["include_source"]
        self.tok = AutoTokenizer.from_pretrained(folder / "tokenizer")
        self.backbone = AutoModel.from_pretrained(folder / "backbone").float().eval()  # funciona también si se guardó en fp16
        self.head = torch.nn.Linear(self.backbone.config.hidden_size, 2)
        self.head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu"))
        self.head.eval()

    def predict(self, text: str, source: str, question: str) -> tuple[float, float]:
        t = " ".join(text.split())
        if self.include_source:
            pair, trunc = f"{question} {self.tok.sep_token} {source}", "only_second"
        else:
            pair, trunc = question, "longest_first"
        enc = self.tok(t, pair or None, truncation=trunc, max_length=self.max_len, return_tensors="pt")
        with self.torch.no_grad():
            h = self.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out = self.head((h * m).sum(1) / m.sum(1).clamp(min=1))[0]
        return float(out[0]), float(out[1])


class Predictor:
    def __init__(self):
        self.sem = joblib.load(MODELS / "semantic.joblib")
        self.cols = joblib.load(MODELS / "columns.joblib")["columns"]
        self.feature_models = {n: joblib.load(MODELS / f"{n}.joblib") for n in FEATURE_MODELS
                               if (MODELS / f"{n}.joblib").exists()}
        mj = RES / "metrics_all.json"
        self.metrics = json.loads(mj.read_text()) if mj.exists() else {}
        self.deberta, self.deberta_name = None, None
        best = self.metrics.get("best_deberta")  # p.ej. "deberta_small"
        for f in (sorted(glob.glob(str(RES / "deberta_*_final"))) + sorted(glob.glob(str(MODELS / "deberta_*_final")))
              + sorted(glob.glob(str(ROOT / "deberta_results" / "deberta_*_final")))):
            tag = Path(f).name.replace("_final", "")
            if best and tag != best:
                continue
            try:
                self.deberta, self.deberta_name = DebertaPredictor(Path(f)), tag
                break
            except Exception as e:  # torch/transformers no instalados o carpeta incompleta
                self.deberta_error = str(e)

    # ---------------------------------------------------------------- rasgos
    def featurize(self, text: str, source: str, question: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
        S = pd.DataFrame({"student_id": ["nuevo"], "prompt_id": ["nuevo"], "text": [text]})
        P = pd.DataFrame({"prompt_id": ["nuevo"], "prompt_question": [question],
                          "prompt_title": ["(texto ingresado)"], "prompt_text": [eu.basic_clean(source)]})
        raw = F.build_raw_features(S, P, self.sem)
        return raw, F.to_model_matrix(raw)[self.cols]

    def available(self) -> list[str]:
        names = list(self.feature_models)
        if self.deberta:
            names.append("deberta")
        ens = self.metrics.get("models", {}).get("ensamble")
        if ens and (self.deberta or not any("deberta" in b for b in ens["bases"])):
            names.append("ensamble")
        return names

    def predict_all(self, text: str, source: str, question: str = "") -> tuple[dict, pd.DataFrame]:
        raw, X = self.featurize(text, source, question)
        out: dict[str, tuple[float, float]] = {}
        for n, spec in self.feature_models.items():
            Xm = X[spec["features"]].to_numpy()
            if spec["kind"] == "lgbm":
                p = np.column_stack([m.predict(Xm) for m in spec["model"]])[0]
            else:
                p = spec["model"].predict(Xm)[0]
            out[n] = (float(p[0]), float(p[1]))
        if self.deberta:
            out["deberta"] = self.deberta.predict(text, source, question)
        ens = self.metrics.get("models", {}).get("ensamble")
        if ens and "ensamble" in self.available():
            vals = {}
            for b in ens["bases"]:
                key = "deberta" if b.startswith("deberta") else b
                vals[f"{b}__content"], vals[f"{b}__wording"] = out[key]
            out["ensamble"] = tuple(
                ens["weights"][t]["intercepto"] + sum(w * vals[c] for c, w in ens["weights"][t].items() if c != "intercepto")
                for t in T)
        return out, X
