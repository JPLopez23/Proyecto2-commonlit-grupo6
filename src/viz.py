"""Paleta y estilo compartidos por las figuras del informe y la app.

Paleta categórica validada con el script del skill `dataviz` (CVD ΔE ≥ 9 entre vecinos,
piso de visión normal ≥ 24). El color sigue a la entidad (modelo), nunca a su ranking.
El aqua queda bajo 3:1 de contraste sobre fondo claro → siempre hay etiquetas de valor visibles.
"""
from __future__ import annotations

import matplotlib as mpl

# Color por modelo (fijo)
COLORS = {
    "baseline_longitud": "#8a8985",   # neutro: es la referencia, no un candidato
    "ridge_enriquecido": "#2a78d6",
    "lgbm_enriquecido": "#eb6834",
    "deberta": "#1baf7a",
    "ensamble": "#6250d6",
}
LABELS = {
    "baseline_longitud": "Línea base (solo longitud)",
    "ridge_enriquecido": "Ridge",
    "lgbm_enriquecido": "LightGBM",
    "deberta": "DeBERTa-v3",
    "ensamble": "Ensamble (stacking)",
}
ORDER = ["baseline_longitud", "ridge_enriquecido", "lgbm_enriquecido", "deberta", "ensamble"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PROMPT_TITLES = {"39c16e": "On Tragedy", "3b9047": "Egyptian Social Structure",
                 "814d6b": "The Third Wave", "ebad26": "Excerpt from The Jungle"}


PROMPT_SHORT = {"39c16e": "Tragedy", "3b9047": "Egypt", "814d6b": "Third Wave", "ebad26": "The Jungle"}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
    })
