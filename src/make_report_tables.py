"""Genera informe/tablas_*.tex y informe/numeros.tex a partir de results/ para que las cifras del
informe coincidan exactamente con las métricas calculadas. Uso: python src/make_report_tables.py"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "informe"
M = json.loads((RES / "metrics_all.json").read_text())
models, best_deb = M["models"], M.get("best_deberta")
PT = {"39c16e": "On Tragedy", "3b9047": "Egyptian Social Structure",
      "814d6b": "The Third Wave", "ebad26": "The Jungle"}
NAMES = {"baseline_longitud": "Línea base (solo longitud)", "ridge_base": "Ridge, 18 rasgos",
         "ridge_enriquecido": "Ridge, 33 rasgos", "ensamble_rasgos": "Ensamble sin DeBERTa (Ridge + LightGBM)", "lgbm_base": "LightGBM, 18 rasgos",
         "lgbm_enriquecido": "LightGBM, 33 rasgos", "ensamble": "Ensamble (stacking)"}


def f(x, d=3):
    return f"{x:.{d}f}".replace(".", "{,}")


def name(k):
    if k.startswith("deberta"):
        return "DeBERTa-v3 " + k.split("_", 1)[1].replace("_", " ")
    return NAMES[k]


order = sorted(models, key=lambda k: models[k]["mcrmse_mean_prompts"])
best = min(models.values(), key=lambda v: v["mcrmse_mean_prompts"])
rows = []
for k in order:
    v = models[k]
    t = v.get("fit_seconds_mean")
    rows.append(f"{name(k)} & {f(v['mcrmse_mean_prompts'])} & {f(v['mcrmse_pooled'])} & {f(v['rmse_content'])} & "
                f"{f(v['rmse_wording'])} & {('--' if t is None or t != t else ('<0{,}1' if t < 0.1 else f(t, 1)))} \\\\")
tab_modelos = ("\\begin{table}[ht]\n\\centering\\small\n"
               "\\caption{Resultados fuera de muestra (Leave-One-Prompt-Out). MCRMSE y RMSE: menor es mejor.}\n\\label{tab:modelos}\n"
               "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccccc}\n\\toprule\nModelo & \\makecell{MCRMSE\\\\(media por texto)} & \\makecell{MCRMSE\\\\(agrupado)} & "
               "\\makecell{RMSE\\\\content} & \\makecell{RMSE\\\\wording} & \\makecell{Segundos\\\\por pliegue} \\\\\n\\midrule\n"
               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

prompts = list(PT)
rows = [f"{name(k)} & " + " & ".join(f(models[k]['per_prompt'][g]) for g in prompts) + " \\\\" for k in order]
tab_prompt = ("\\begin{table}[ht]\n\\centering\\small\n\\caption{MCRMSE según el texto fuente que se dejó fuera de entrenamiento.}\n"
              "\\label{tab:porprompt}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccc}\n\\toprule\nModelo & "
              + " & ".join(PT[g] for g in prompts) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

rows = []
for k in ["ridge_base", "ridge_enriquecido", "lgbm_base", "lgbm_enriquecido"]:
    cp = models[k]["chosen_params"]
    desc = []
    for g in prompts:
        p = cp[g]
        desc.append(f"$\\alpha={p['alpha']:g}$" if "alpha" in p else f"{p['num_leaves']} hojas, {p['n_estimators']} árb.")
    rows.append(f"{name(k)} & " + " & ".join(desc) + " \\\\")
tab_params = ("\\begin{table}[ht]\n\\centering\\footnotesize\n\\caption{Hiperparámetros elegidos en la validación interna, según el texto que se deja fuera.}\n"
              "\\label{tab:params}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccc}\n\\toprule\nModelo & " + " & ".join(PT[g] for g in prompts) +
              " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
(OUT / "tablas_modelos.tex").write_text(tab_modelos)
(OUT / "tablas_porprompt.tex").write_text(tab_prompt)
(OUT / "tablas_params.tex").write_text(tab_params)

macros = {}
for k, v in models.items():
    key = k.replace("_", "").replace("ñ", "n")
    macros[f"mc{key}"] = f(v["mcrmse_mean_prompts"])
    macros[f"rc{key}"] = f(v["rmse_content"])
    macros[f"rw{key}"] = f(v["rmse_wording"])
(OUT / "numeros.tex").write_text("".join(
    "\\newcommand{\\" + "".join(c for c in k if c.isalpha()) + "}{" + v + "}\n" for k, v in macros.items()))
print(sorted(macros)[:40])
