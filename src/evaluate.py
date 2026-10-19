"""Une las predicciones fuera de muestra (rasgos + DeBERTa), construye el ensamble por
stacking con validación Leave-One-Prompt-Out, calcula métricas y genera las figuras.

Uso: python src/evaluate.py
Entradas: results/oof_cpu.csv, results/metrics_cpu.json y (opcional) results/deberta_*_oof.csv
Salidas: results/metrics_all.json, results/tabla_modelos.csv, figuras/fig_*.png
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import viz  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figuras"
FIG.mkdir(exist_ok=True)
T = ["content", "wording"]


def mcrmse(y, p):
    return float(np.mean(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2, axis=0))))


def pred_cols(name):
    return [f"{name}__content", f"{name}__wording"]


def stack_lopo(oof: pd.DataFrame, bases: list[str]):
    """Meta-modelo Ridge (coeficientes ≥ 0) entrenado con las predicciones fuera de muestra de
    los otros 3 textos; se evalúa en el texto restante."""
    Xcols = [c for b in bases for c in pred_cols(b)]
    out = np.zeros((len(oof), 2))
    for g in oof["prompt_id"].unique():
        tr, te = oof["prompt_id"] != g, oof["prompt_id"] == g
        for j, t in enumerate(T):
            m = Ridge(alpha=1.0, positive=True).fit(oof.loc[tr, Xcols], oof.loc[tr, t])
            out[te.to_numpy(), j] = m.predict(oof.loc[te, Xcols])
    final = [Ridge(alpha=1.0, positive=True).fit(oof[Xcols], oof[t]) for t in T]
    weights = {t: dict(zip(Xcols, np.round(m.coef_, 3))) | {"intercepto": round(float(m.intercept_), 3)}
               for t, m in zip(T, final)}
    return out, weights, Xcols


def main():
    oof = pd.read_csv(RES / "oof_cpu.csv")
    mc = json.loads((RES / "metrics_cpu.json").read_text())["models"]
    metrics = {k: dict(v) for k, v in mc.items()}

    # ---- DeBERTa (opcional): se elige la variante con menor MCRMSE agrupado
    deb = {}
    for f in sorted(glob.glob(str(RES / "deberta_*_oof.csv"))):
        tag = Path(f).stem.replace("deberta_", "").replace("_oof", "")
        d = pd.read_csv(f)
        c = [f"deberta_{tag}__content", f"deberta_{tag}__wording"]
        if d[c].isna().any().any():
            continue
        oof = oof.merge(d[["student_id"] + c], on="student_id", how="left")
        mj = RES / f"deberta_{tag}_metrics.json"
        info = json.loads(mj.read_text()) if mj.exists() else {}
        pp = {g: mcrmse(oof.loc[oof.prompt_id == g, T], oof.loc[oof.prompt_id == g, c]) for g in oof.prompt_id.unique()}
        name = f"deberta_{tag}"
        metrics[name] = {
            "mcrmse_mean_prompts": float(np.mean(list(pp.values()))),
            "mcrmse_pooled": mcrmse(oof[T], oof[c]),
            "rmse_content": float(np.sqrt(np.mean((oof["content"] - oof[c[0]]) ** 2))),
            "rmse_wording": float(np.sqrt(np.mean((oof["wording"] - oof[c[1]]) ** 2))),
            "per_prompt": pp,
            "fit_seconds_mean": float(np.mean(list(info.get("fold_seconds", {}).values()) or [np.nan])),
            "device_note": "GPU (Kaggle/Colab)",
        }
        deb[tag] = name
    best_deb = min(deb.values(), key=lambda n: metrics[n]["mcrmse_mean_prompts"]) if deb else None

    # ---- Ensamble
    bases = ["ridge_enriquecido", "lgbm_enriquecido"] + ([best_deb] if best_deb else [])
    pred, weights, _ = stack_lopo(oof, bases)
    for j, t in enumerate(T):
        oof[f"ensamble__{t}"] = pred[:, j]
    pp = {g: mcrmse(oof.loc[oof.prompt_id == g, T], oof.loc[oof.prompt_id == g, pred_cols("ensamble")])
          for g in oof.prompt_id.unique()}
    metrics["ensamble"] = {
        "mcrmse_mean_prompts": float(np.mean(list(pp.values()))),
        "mcrmse_pooled": mcrmse(oof[T], oof[pred_cols("ensamble")]),
        "rmse_content": float(np.sqrt(np.mean((oof["content"] - oof["ensamble__content"]) ** 2))),
        "rmse_wording": float(np.sqrt(np.mean((oof["wording"] - oof["ensamble__wording"]) ** 2))),
        "per_prompt": pp, "bases": bases, "weights": weights,
        "fit_seconds_mean": 0.01,
    }
    # Referencia: ensamble solo con modelos de rasgos (sin DeBERTa)
    if best_deb:
        pr, _, _ = stack_lopo(oof, ["ridge_enriquecido", "lgbm_enriquecido"])
        pc = ["ensamble_rasgos__content", "ensamble_rasgos__wording"]
        oof[pc[0]], oof[pc[1]] = pr[:, 0], pr[:, 1]
        ppr = {g: mcrmse(oof.loc[oof.prompt_id == g, T], oof.loc[oof.prompt_id == g, pc]) for g in oof.prompt_id.unique()}
        metrics["ensamble_rasgos"] = {
            "mcrmse_mean_prompts": float(np.mean(list(ppr.values()))), "mcrmse_pooled": mcrmse(oof[T], oof[pc]),
            "rmse_content": float(np.sqrt(np.mean((oof["content"] - oof[pc[0]]) ** 2))),
            "rmse_wording": float(np.sqrt(np.mean((oof["wording"] - oof[pc[1]]) ** 2))),
            "per_prompt": ppr, "fit_seconds_mean": 0.01}
    oof.to_csv(RES / "oof_all.csv", index=False)
    (RES / "metrics_all.json").write_text(json.dumps(
        {"models": metrics, "best_deberta": best_deb}, indent=2, default=str))

    # ---- Tabla resumen
    rows = []
    for k, v in metrics.items():
        rows.append({"modelo": k, "MCRMSE (media por texto)": v["mcrmse_mean_prompts"],
                     "MCRMSE (agrupado)": v["mcrmse_pooled"], "RMSE content": v["rmse_content"],
                     "RMSE wording": v["rmse_wording"], **{f"MCRMSE {viz.PROMPT_TITLES[g]}": s for g, s in v["per_prompt"].items()},
                     "segundos de ajuste (por pliegue)": v.get("fit_seconds_mean")})
    tab = pd.DataFrame(rows).sort_values("MCRMSE (media por texto)")
    tab.to_csv(RES / "tabla_modelos.csv", index=False)
    print(tab[["modelo", "MCRMSE (media por texto)", "MCRMSE (agrupado)", "RMSE content", "RMSE wording"]].round(4).to_string(index=False))

    make_figures(oof, metrics, best_deb)


# ------------------------------------------------------------------- figuras
def key_for(k, best_deb):
    return "deberta" if k == best_deb else k


def make_figures(oof, metrics, best_deb):
    viz.apply_style()
    main_models = ["baseline_longitud", "ridge_enriquecido", "lgbm_enriquecido"] + \
                  ([best_deb] if best_deb else []) + ["ensamble"]
    col = {m: viz.COLORS[key_for(m, best_deb)] for m in main_models}
    lab = {m: (viz.LABELS[key_for(m, best_deb)] + (f" ({m.split('_',1)[1]})" if m == best_deb else "")) for m in main_models}
    prompts = list(viz.PROMPT_TITLES)

    # 1. MCRMSE global por modelo
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    vals = [metrics[m]["mcrmse_mean_prompts"] for m in main_models]
    bars = ax.barh([lab[m] for m in main_models][::-1], vals[::-1], color=[col[m] for m in main_models][::-1], height=0.62)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", color=viz.INK, fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("MCRMSE (menor es mejor) — media de los 4 pliegues, un texto fuente fuera en cada uno")
    ax.grid(axis="y", visible=False); ax.grid(axis="x")
    ax.set_title("Error de cada modelo en textos fuente que no vio")
    fig.tight_layout(); fig.savefig(FIG / "fig_mcrmse_modelos.png"); plt.close(fig)

    # 2. MCRMSE por texto
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    w = 0.8 / len(main_models); x = np.arange(len(prompts))
    for i, m in enumerate(main_models):
        v = [metrics[m]["per_prompt"][g] for g in prompts]
        r = ax.bar(x + (i - (len(main_models) - 1) / 2) * w, v, w * 0.92, color=col[m], label=lab[m])
        for rect, val in zip(r, v):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7.5, color=viz.INK2)
    ax.set_xticks(x, [viz.PROMPT_TITLES[g].replace("Excerpt from ", "") for g in prompts])
    ax.set_ylabel("MCRMSE"); ax.set_title("El error depende del texto que se deja fuera")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_mcrmse_por_texto.png"); plt.close(fig)

    # 3. Predicho vs. real del mejor modelo
    best = min([m for m in main_models if m != "baseline_longitud"], key=lambda m: metrics[m]["mcrmse_mean_prompts"])
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for a, t in zip(axs, T):
        p = oof[f"{best}__{t}"]
        a.scatter(oof[t], p, s=7, alpha=0.28, color=col[best], linewidths=0)
        lo, hi = min(oof[t].min(), p.min()), max(oof[t].max(), p.max())
        a.plot([lo, hi], [lo, hi], color=viz.INK2, lw=1, ls="--")
        a.set_xlabel(f"{t} real"); a.set_ylabel(f"{t} predicho")
        r = np.corrcoef(oof[t], p)[0, 1]
        a.set_title(f"{t}  (r = {r:.2f})"); a.grid(axis="both")
    fig.suptitle(f"Predicción fuera de muestra: {lab[best]}", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig_pred_vs_real.png"); plt.close(fig)

    # 4. Error por cuantil de longitud
    q = pd.qcut(oof["word_count"], 5, labels=False, duplicates="drop")
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    centers = oof.groupby(q)["word_count"].median().astype(int)
    for m in main_models:
        e = [mcrmse(oof.loc[q == k, T], oof.loc[q == k, pred_cols(m)]) for k in sorted(q.unique())]
        ax.plot(range(len(e)), e, marker="o", ms=6, lw=2, color=col[m], label=lab[m])
    ax.set_xticks(range(len(centers)), [f"Q{i+1}\n(~{c} palabras)" for i, c in enumerate(centers)])
    ax.set_ylabel("MCRMSE"); ax.set_title("¿Funciona también en resúmenes cortos y largos?")
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_error_por_longitud.png"); plt.close(fig)

    # 5. Importancia de rasgos (LightGBM)
    imp = pd.read_csv(RES / "importancia_lgbm.csv", index_col=0)
    imp["total"] = imp.mean(axis=1)
    top = imp.sort_values("total", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(top)); h = 0.38
    ax.barh(y + h / 2, top["content"], h * 0.92, color=viz.COLORS["lgbm_enriquecido"], label="content")
    ax.barh(y - h / 2, top["wording"], h * 0.92, color=viz.SEQ_BLUE[5], label="wording")
    ax.set_yticks(y, top.index); ax.grid(axis="y", visible=False); ax.grid(axis="x")
    ax.set_xlabel("Importancia relativa (ganancia)"); ax.set_title("Qué rasgos usa LightGBM para cada nota")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig_importancia_lightgbm.png"); plt.close(fig)

    # 6. Tiempo de ajuste
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ms = [m for m in main_models if m not in ("baseline_longitud", "ensamble")]
    t = [max(metrics[m].get("fit_seconds_mean") or 0.01, 0.01) for m in ms]
    b = ax.barh([lab[m] for m in ms][::-1], t[::-1], color=[col[m] for m in ms][::-1], height=0.55)
    for rect, v in zip(b, t[::-1]):
        ax.text(v * 1.08, rect.get_y() + rect.get_height() / 2, (f"{v:.1f} s" if v >= 0.1 else "< 0.1 s"), va="center", fontweight="bold")
    ax.set_xscale("log"); ax.grid(axis="y", visible=False); ax.grid(axis="x")
    ax.set_xlabel("Segundos por pliegue (escala logarítmica)")
    ax.set_title("Costo de entrenamiento"); fig.tight_layout(); fig.savefig(FIG / "fig_tiempos.png"); plt.close(fig)
    print("figuras:", sorted(p.name for p in FIG.glob("fig_*.png")))


if __name__ == "__main__":
    main()
