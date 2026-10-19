"""Entrena y evalúa los modelos basados en rasgos (Ridge, LightGBM) con validación
Leave-One-Prompt-Out (equivale a GroupKFold por prompt_id con 4 grupos).

Para cada texto fuente de prueba se ajustan los hiperparámetros solo con los otros tres
(validación anidada), de modo que el texto evaluado nunca influye en su propia selección.

Uso:  python src/train_cpu.py
Salida: results/oof_cpu.csv, results/metrics_cpu.json, models/*.joblib, data/features_*.parquet
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eda_utils as eu  # noqa: E402
import features as F  # noqa: E402

DATA, RES, MODELS = ROOT / "data", ROOT / "results", ROOT / "models"
RES.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
TARGETS = eu.TARGETS
SEED = 42


def mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))))


# ------------------------------------------------------------------ modelos
class RidgeModel:
    name = "ridge"
    grid = [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0)]

    def make(self, p):
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))

    def fit(self, p, X, Y):
        return self.make(p).fit(X, Y)

    def predict(self, m, X):
        return m.predict(X)


class LGBMModel:
    """Un regresor por nota (las dos notas se predicen por separado)."""
    name = "lgbm"
    grid = [
        {"num_leaves": nl, "n_estimators": ne, "learning_rate": 0.03, "min_child_samples": mcs}
        for nl, ne, mcs in itertools.product((7, 15), (250, 500), (60,))
    ]

    def fit(self, p, X, Y):
        ms = []
        for j in range(Y.shape[1]):
            m = lgb.LGBMRegressor(
                objective="regression", subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=5.0, random_state=SEED, n_jobs=2, verbose=-1, **p)
            ms.append(m.fit(X, Y[:, j]))
        return ms

    def predict(self, ms, X):
        return np.column_stack([m.predict(X) for m in ms])


SPECS = {
    # nombre -> (clase, conjunto de características)
    "baseline_longitud": (RidgeModel, "length"),
    "ridge_base": (RidgeModel, "base21"),
    "ridge_enriquecido": (RidgeModel, "enriquecido"),
    "lgbm_base": (LGBMModel, "base21"),
    "lgbm_enriquecido": (LGBMModel, "enriquecido"),
}


def select_params(model, X, Y, groups):
    """LOPO interno: devuelve el hiperparámetro con menor MCRMSE medio."""
    best, best_s = None, np.inf
    if len(model.grid) == 1:
        return model.grid[0]
    for p in model.grid:
        scores = []
        for g in np.unique(groups):
            tr, va = groups != g, groups == g
            m = model.fit(p, X[tr], Y[tr])
            scores.append(mcrmse(Y[va], model.predict(m, X[va])))
        s = float(np.mean(scores))
        if s < best_s:
            best, best_s = p, s
    return best


def main() -> None:
    t0 = time.time()
    S = pd.read_csv(DATA / "summaries_train.csv")
    P = pd.read_csv(DATA / "prompts_train.csv")
    S["text"] = S["text"].map(eu.basic_clean)

    # Modelo semántico no supervisado (TF-IDF + LSA): usa solo texto, nunca las notas.
    corpus = S["text"].tolist() + P["prompt_text"].map(eu.basic_clean).tolist()
    sem = F.SemanticModel().fit(corpus)
    raw = F.build_raw_features(S, P, sem)
    X_all = F.to_model_matrix(raw)
    sets = F.feature_sets(list(X_all.columns))
    sets["length"] = F.length_only(list(X_all.columns))
    Y = raw[TARGETS].to_numpy()
    groups = raw["prompt_id"].to_numpy()
    prompts = sorted(np.unique(groups))
    print(f"características: {X_all.shape[1]}  (base: {len(sets['base21'])}, "
          f"enriquecido: {len(sets['enriquecido'])})  [{time.time()-t0:.0f}s]")

    X_all.assign(prompt_id=groups, student_id=raw["student_id"].to_numpy(),
                 content=Y[:, 0], wording=Y[:, 1]).to_parquet(DATA / "features_model.parquet", index=False)

    oof = pd.DataFrame({"student_id": raw["student_id"], "prompt_id": groups,
                        "content": Y[:, 0], "wording": Y[:, 1],
                        "word_count": raw["word_count"]})
    metrics: dict = {"models": {}, "feature_sets": {k: v for k, v in sets.items()}}

    for name, (cls, fs) in SPECS.items():
        model = cls()
        X = X_all[sets[fs]].to_numpy()
        pred = np.zeros_like(Y)
        per_prompt, chosen, fit_times = {}, {}, []
        for g in prompts:
            te, tr = groups == g, groups != g
            params = select_params(model, X[tr], Y[tr], groups[tr])
            t1 = time.time()
            m = model.fit(params, X[tr], Y[tr])
            fit_times.append(time.time() - t1)
            pred[te] = model.predict(m, X[te])
            per_prompt[g] = mcrmse(Y[te], pred[te])
            chosen[g] = params
        for j, t in enumerate(TARGETS):
            oof[f"{name}__{t}"] = pred[:, j]
        metrics["models"][name] = {
            "mcrmse_mean_prompts": float(np.mean(list(per_prompt.values()))),
            "mcrmse_pooled": mcrmse(Y, pred),
            "rmse_content": float(np.sqrt(np.mean((Y[:, 0] - pred[:, 0]) ** 2))),
            "rmse_wording": float(np.sqrt(np.mean((Y[:, 1] - pred[:, 1]) ** 2))),
            "per_prompt": per_prompt,
            "chosen_params": chosen,
            "fit_seconds_mean": float(np.mean(fit_times)),
        }
        print(f"{name:20s} MCRMSE(media por texto)={metrics['models'][name]['mcrmse_mean_prompts']:.4f} "
              f"pooled={metrics['models'][name]['mcrmse_pooled']:.4f}  [{time.time()-t0:.0f}s]")

    # ---- modelos finales (todos los textos) para la app
    for name, (cls, fs) in SPECS.items():
        model = cls()
        X = X_all[sets[fs]].to_numpy()
        params = select_params(model, X, Y, groups)
        joblib.dump({"model": model.fit(params, X, Y), "features": sets[fs], "kind": cls.name,
                     "params": params}, MODELS / f"{name}.joblib")
    joblib.dump(sem, MODELS / "semantic.joblib")
    joblib.dump({"columns": list(X_all.columns)}, MODELS / "columns.joblib")

    # importancias del LGBM final
    lg = joblib.load(MODELS / "lgbm_enriquecido.joblib")
    imp = pd.DataFrame({t: m.booster_.feature_importance("gain") for t, m in zip(TARGETS, lg["model"])},
                       index=lg["features"])
    imp = imp / imp.sum()
    imp.to_csv(RES / "importancia_lgbm.csv")
    rid = joblib.load(MODELS / "ridge_enriquecido.joblib")["model"]
    pd.DataFrame(rid[-1].coef_.T, index=sets["enriquecido"], columns=TARGETS).to_csv(RES / "coef_ridge.csv")

    oof.to_csv(RES / "oof_cpu.csv", index=False)
    (RES / "metrics_cpu.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"listo en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
