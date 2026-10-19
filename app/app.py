"""CommonLit · Evaluador de resúmenes — Grupo 6 (CC3084, UVG).

Ejecutar desde la raíz del repositorio:   streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import viz  # noqa: E402
from inference import Predictor  # noqa: E402

T = ["content", "wording"]
RES, DATA = ROOT / "results", ROOT / "data"
FEATURE_ES = {
    "log_word_count": "Palabras (log)", "log_char_count": "Caracteres (log)",
    "log_sentence_count": "Oraciones (log)", "log_unique_word_count": "Palabras únicas (log)",
    "log_comma_count": "Comas (log)", "guiraud": "Diversidad léxica ajustada (Guiraud)",
    "avg_word_length": "Longitud media de palabra", "avg_sentence_length_words": "Longitud media de oración",
    "stopword_ratio": "Proporción de palabras vacías", "punct_per_word": "Puntuación por palabra",
    "uppercase_ratio": "Proporción de mayúsculas", "jaccard_unigram": "Similitud de Jaccard con la fuente",
    "prop_words_in_source": "Palabras del resumen que están en la fuente",
    "bigram_overlap_ratio": "Solapamiento de bigramas", "trigram_overlap_ratio": "Solapamiento de trigramas",
    "new_word_ratio": "Palabras nuevas (no están en la fuente)", "flesch_reading_ease": "Legibilidad Flesch",
    "fk_grade": "Grado Flesch-Kincaid", "gunning_fog": "Índice Gunning Fog", "ari": "Índice ARI",
    "syll_per_word": "Sílabas por palabra", "complex_word_ratio": "Proporción de palabras complejas",
    "misspelled_ratio": "Proporción de errores de ortografía", "log_misspelled_count": "Errores de ortografía (log)",
    "tfidf_cos_source": "Similitud TF-IDF con la fuente", "lsa_cos_source": "Similitud semántica (LSA) con la fuente",
    "tfidf_cos_question": "Similitud con la consigna", "cov_max_sim": "Cobertura: mejor oración de la fuente",
    "cov_mean_sim": "Cobertura: similitud media", "cov_frac_sentences": "Cobertura: fracción de oraciones tocadas",
    "cov_top3_mean": "Cobertura: promedio de las 3 mejores", "flag_very_short": "Bandera: resumen muy corto",
    "flag_very_long": "Bandera: resumen muy largo",
}

st.set_page_config(page_title="Evaluador de resúmenes · CommonLit", page_icon="📝", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1 {font-size: 1.9rem !important; letter-spacing: -0.01em;}
.card {background:#f3f2ee; border-radius:12px; padding:14px 16px; border-left:6px solid var(--c,#2a78d6);}
.card .t {font-size:.8rem; color:#52514e; margin-bottom:2px;}
.card .v {font-size:1.7rem; font-weight:700; color:#0b0b0b; line-height:1.15;}
.card .s {font-size:.8rem; color:#52514e;}
.note {color:#52514e; font-size:.9rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------- datos
@st.cache_resource(show_spinner="Cargando modelos…")
def load_predictor():
    return Predictor()


@st.cache_data
def load_prompts():
    return pd.read_csv(DATA / "prompts_train.csv").set_index("prompt_id")


@st.cache_data
def load_features():
    p = DATA / "features_model.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["prompt"] = d["prompt_id"].map(viz.PROMPT_TITLES)
    return d


@st.cache_data
def load_results():
    mj, oj = RES / "metrics_all.json", RES / "oof_all.csv"
    if not (mj.exists() and oj.exists()):
        return None, None
    return json.loads(mj.read_text()), pd.read_csv(oj)


pred = load_predictor()
prompts = load_prompts()
feats = load_features()
metrics_all, oof = load_results()
best_deb = (metrics_all or {}).get("best_deberta")


def mkey(name: str) -> str:
    """Nombre interno del modelo → clave de color/etiqueta."""
    return "deberta" if name.startswith("deberta") else name


def mlabel(name: str) -> str:
    return viz.LABELS[mkey(name)]


def mcolor(name: str) -> str:
    return viz.COLORS[mkey(name)]


def plot_layout(fig, h=380, **kw):
    fig.update_layout(template="plotly_white", height=h, paper_bgcolor=viz.SURFACE, plot_bgcolor=viz.SURFACE,
                      font=dict(color=viz.INK2, size=13), margin=dict(l=10, r=10, t=50, b=10),
                      title=dict(font=dict(size=16, color=viz.INK), x=0), legend=dict(orientation="h", y=-0.18), **kw)
    fig.update_xaxes(gridcolor=viz.GRID, zerolinecolor=viz.GRID)
    fig.update_yaxes(gridcolor=viz.GRID, zerolinecolor=viz.GRID)
    return fig


def card(title, value, sub, color):
    st.markdown(f'<div class="card" style="--c:{color}"><div class="t">{title}</div>'
                f'<div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------- sidebar
st.title("Evaluador automático de resúmenes")
st.markdown('<p class="note">Reto CommonLit · Grupo 6 · CC3084 Data Science, UVG. Predice dos notas de un experto: '
            '<b>content</b> (cobertura de las ideas del texto) y <b>wording</b> (calidad de la redacción). '
            'Las notas están estandarizadas: 0 es el promedio, +1 está una desviación por encima.</p>',
            unsafe_allow_html=True)
avail = pred.available()
with st.sidebar:
    st.header("Controles")
    chosen = st.multiselect("Modelos a usar", avail, default=avail, format_func=mlabel,
                            help="Puedes ver el resultado de cada algoritmo por separado o de todos a la vez.")
    show_perf = st.toggle("Mostrar rendimiento de los modelos", value=True,
                          help="Oculta o muestra la pestaña con las gráficas de rendimiento.")
    st.divider()
    st.caption("Los modelos nunca vieron el texto fuente que se usa para validar: se evaluaron dejando un "
               "texto fuente fuera en cada pliegue.")
    if not pred.deberta:
        st.info("DeBERTa no está cargado. Coloca la carpeta `deberta_*_final` generada por el notebook de GPU en "
                "`results/` para activarlo.")

tab_names = ["📝 Calificar un resumen", "🔎 Explorar los datos"] + (["📊 Rendimiento de los modelos"] if show_perf else [])
tabs = st.tabs(tab_names)

# ---------------------------------------------------------- 1. Calificar
with tabs[0]:
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        src_mode = st.radio("Texto fuente", ["Uno de los 4 textos del curso", "Mi propio texto fuente"], horizontal=True)
        question = ""
        if src_mode.startswith("Uno"):
            pid = st.selectbox("Texto", list(prompts.index), format_func=lambda p: prompts.loc[p, "prompt_title"])
            source, question = prompts.loc[pid, "prompt_text"], prompts.loc[pid, "prompt_question"]
            st.caption(f"Consigna: {question}")
            with st.expander("Ver el texto fuente"):
                st.write(source)
        else:
            pid = None
            question = st.text_input("Consigna (opcional)", "Summarize the main ideas of the text.")
            source = st.text_area("Pega aquí el texto fuente (en inglés)", height=200)
    with c2:
        if "summary" not in st.session_state:
            st.session_state.summary = ""
        if pid is not None and st.button("Cargar un resumen real de ejemplo"):
            ex = pd.read_csv(DATA / "summaries_train.csv").query("prompt_id == @pid").sample(1).iloc[0]
            st.session_state.summary = ex["text"]
            st.session_state.real = (ex["content"], ex["wording"])
        summary = st.text_area("Resumen del estudiante (en inglés)", key="summary", height=250,
                               placeholder="Escribe o pega un resumen…")
        go_btn = st.button("Calificar", type="primary", disabled=not (summary.strip() and source.strip() and chosen))

    if go_btn:
        with st.spinner("Calculando…"):
            preds, X = pred.predict_all(summary, source, question)
        st.divider()
        st.subheader("Resultados por modelo")
        cols = st.columns(len(chosen))
        for col, m in zip(cols, chosen):
            with col:
                card(mlabel(m), f"{preds[m][0]:+.2f} · {preds[m][1]:+.2f}", "content · wording", mcolor(m))
        if "real" in st.session_state and pid is not None and summary == st.session_state.get("summary") and st.session_state.get("real"):
            st.caption(f"Notas reales del experto en este resumen de ejemplo: content {st.session_state.real[0]:+.2f} · "
                       f"wording {st.session_state.real[1]:+.2f}")
        fig = go.Figure()
        for t, dash in zip(T, [None, None]):
            fig.add_bar(name=t, x=[mlabel(m) for m in chosen], y=[preds[m][T.index(t)] for m in chosen],
                        text=[f"{preds[m][T.index(t)]:+.2f}" for m in chosen], textposition="outside",
                        marker=dict(color=[mcolor(m) for m in chosen], pattern_shape="" if t == "content" else "/",
                                    line=dict(color=viz.SURFACE, width=2)),
                        hovertemplate="%{x}<br>" + t + ": %{y:+.2f}<extra></extra>")
        fig.update_layout(barmode="group", title="Notas predichas (relleno liso = content, rayado = wording)")
        fig.add_hline(y=0, line_color=viz.MUTED, line_width=1)
        st.plotly_chart(plot_layout(fig, 360, showlegend=False), width="stretch")

        # ---- diagnóstico: el resumen frente a los datos
        _, Xn = pred.featurize(summary, source, question)
        row = Xn.iloc[0]
        w = int(round(np.expm1(row["log_word_count"])))
        msgs = []
        if row["flag_very_short"]:
            msgs.append(f"El resumen es muy corto ({w} palabras): la longitud es la señal más fuerte y las notas tienden a ser bajas.")
        if row["flag_very_long"]:
            msgs.append(f"El resumen es muy largo ({w} palabras): suele cubrir más contenido, pero revisa que no sea copia.")
        if row["trigram_overlap_ratio"] > 0.5:
            msgs.append("Más de la mitad de las frases coinciden con el texto fuente: copiar y pegar penaliza sobre todo wording.")
        if row["misspelled_ratio"] > 0.08:
            msgs.append("Hay una proporción alta de palabras que el corrector no reconoce.")
        for m in msgs:
            st.warning(m)
        if feats is not None:
            st.subheader("Tu resumen frente a los 7 165 resúmenes del entrenamiento")
            keys = ["log_word_count", "prop_words_in_source", "trigram_overlap_ratio", "misspelled_ratio",
                    "lsa_cos_source", "cov_frac_sentences"]
            pct = [float((feats[k] <= row[k]).mean() * 100) for k in keys]
            fig2 = go.Figure(go.Bar(y=[FEATURE_ES[k] for k in keys][::-1], x=pct[::-1], orientation="h",
                                    marker=dict(color=viz.SEQ_BLUE[4], line=dict(color=viz.SURFACE, width=2)),
                                    text=[f"{p:.0f}" for p in pct][::-1], textposition="outside",
                                    hovertemplate="%{y}<br>percentil %{x:.0f}<extra></extra>"))
            fig2.update_xaxes(range=[0, 108], title="Percentil (50 = un resumen típico)")
            fig2.add_vline(x=50, line_dash="dot", line_color=viz.MUTED)
            st.plotly_chart(plot_layout(fig2, 330, showlegend=False), width="stretch")
            st.caption("El texto se limpia y se convierten sus rasgos de forma automática; aquí ves cómo se posiciona frente a los datos.")

# ------------------------------------------------------------ 2. Explorar
with tabs[1]:
    if feats is None:
        st.error("Falta `data/features_model.parquet`. Genera los rasgos con `python src/train_cpu.py`.")
    else:
        st.markdown('<p class="note">Variables del conjunto de datos <b>después de la limpieza</b>, incluidas las que usan '
                    'los modelos. Elige una variable y compárala con las notas.</p>', unsafe_allow_html=True)
        vars_ = [c for c in feats.columns if c in FEATURE_ES]
        a, b, c = st.columns([2, 1, 1])
        var = a.selectbox("Variable", vars_, format_func=lambda v: FEATURE_ES[v], index=vars_.index("log_word_count"))
        target = b.selectbox("Nota", T)
        by_prompt = c.toggle("Separar por texto fuente", value=True)
        colmap = dict(zip(viz.PROMPT_TITLES.values(), ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))

        l, r = st.columns(2, gap="large")
        with l:
            fig = go.Figure()
            groups = list(viz.PROMPT_TITLES.values()) if by_prompt else [None]
            for g in groups:
                d = feats if g is None else feats[feats["prompt"] == g]
                fig.add_histogram(x=d[var], name=g or "Todos", opacity=0.62 if by_prompt else 0.9, nbinsx=40,
                                  marker_color=colmap.get(g, viz.SEQ_BLUE[4]))
            fig.update_layout(barmode="overlay", title=f"Distribución: {FEATURE_ES[var]}")
            st.plotly_chart(plot_layout(fig, 360, showlegend=by_prompt), width="stretch")
        with r:
            fig = go.Figure()
            for g in groups:
                d = feats if g is None else feats[feats["prompt"] == g]
                fig.add_scatter(x=d[var], y=d[target], mode="markers", name=g or "Todos",
                                marker=dict(size=4.5, opacity=0.35, color=colmap.get(g, viz.SEQ_BLUE[4])),
                                hovertemplate=f"{FEATURE_ES[var]}: %{{x:.2f}}<br>{target}: %{{y:.2f}}<extra></extra>")
            q = pd.qcut(feats[var], 12, duplicates="drop")
            mm = feats.groupby(q, observed=True).agg(x=(var, "mean"), y=(target, "mean"))
            fig.add_scatter(x=mm["x"], y=mm["y"], mode="lines+markers", name="Promedio por tramo",
                            line=dict(color=viz.INK, width=2.5), marker=dict(size=8, line=dict(color=viz.SURFACE, width=2)))
            fig.update_layout(title=f"{target} vs. {FEATURE_ES[var]}")
            st.plotly_chart(plot_layout(fig, 360), width="stretch")

        st.subheader("Qué variables se relacionan más con cada nota")
        sp = feats[vars_ + T].corr(method="spearman")[T].drop(T)
        sp = sp.reindex(sp[target].abs().sort_values().index)
        fig = go.Figure(go.Bar(y=[FEATURE_ES[k] for k in sp.index], x=sp[target], orientation="h",
                               marker=dict(color=[viz.COLORS["lgbm_enriquecido"] if v < 0 else viz.SEQ_BLUE[4] for v in sp[target]],
                                           line=dict(color=viz.SURFACE, width=1)),
                               hovertemplate="%{y}<br>Spearman %{x:.2f}<extra></extra>"))
        fig.update_layout(title=f"Correlación de Spearman con {target} (azul positiva, naranja negativa)")
        st.plotly_chart(plot_layout(fig, 760, showlegend=False), width="stretch")
        with st.expander("Estadística descriptiva"):
            st.dataframe(feats[[var] + T].describe().T.round(3), width="stretch")

# ---------------------------------------------------------- 3. Rendimiento
if show_perf:
    with tabs[2]:
        if metrics_all is None:
            st.error("Falta `results/metrics_all.json`. Ejecuta `python src/evaluate.py`.")
        else:
            M = metrics_all["models"]
            mods = [m for m in ["baseline_longitud", "ridge_enriquecido", "lgbm_enriquecido", best_deb, "ensamble"]
                    if m and m in M and (mkey(m) in chosen or mkey(m) not in avail)]
            st.markdown('<p class="note">Todas las cifras son <b>fuera de muestra</b>: cada texto fuente se dejó fuera '
                        'una vez y los modelos se entrenaron con los otros tres. MCRMSE es el error de la competencia '
                        '(menor es mejor).</p>', unsafe_allow_html=True)
            cols = st.columns(len(mods))
            for col, m in zip(cols, mods):
                with col:
                    card(mlabel(m), f"{M[m]['mcrmse_mean_prompts']:.3f}", "MCRMSE medio", mcolor(m))
            order = sorted(mods, key=lambda m: M[m]["mcrmse_mean_prompts"])
            l, r = st.columns(2, gap="large")
            with l:
                fig = go.Figure(go.Bar(y=[mlabel(m) for m in order][::-1], x=[M[m]["mcrmse_mean_prompts"] for m in order][::-1],
                                       orientation="h", marker=dict(color=[mcolor(m) for m in order][::-1], line=dict(color=viz.SURFACE, width=2)),
                                       text=[f"{M[m]['mcrmse_mean_prompts']:.3f}" for m in order][::-1], textposition="outside",
                                       hovertemplate="%{y}<br>MCRMSE %{x:.3f}<extra></extra>"))
                fig.update_layout(title="Error global por modelo")
                fig.update_xaxes(range=[0, max(M[m]["mcrmse_mean_prompts"] for m in mods) * 1.18])
                st.plotly_chart(plot_layout(fig, 360, showlegend=False), width="stretch")
            with r:
                fig = go.Figure()
                for m in mods:
                    fig.add_bar(name=mlabel(m), x=[viz.PROMPT_SHORT[g] for g in viz.PROMPT_TITLES],
                                y=[M[m]["per_prompt"][g] for g in viz.PROMPT_TITLES], marker=dict(color=mcolor(m), line=dict(color=viz.SURFACE, width=2)),
                                hovertemplate="%{x}<br>" + mlabel(m) + ": %{y:.3f}<extra></extra>")
                fig.update_layout(barmode="group", title="Error según el texto que se dejó fuera")
                st.plotly_chart(plot_layout(fig, 360), width="stretch")

            l, r = st.columns(2, gap="large")
            with l:
                sel = st.selectbox("Modelo para comparar predicho vs. real", mods, format_func=mlabel, index=min(2, len(mods) - 1))
                tgt = st.radio("Nota", T, horizontal=True)
                fig = go.Figure()
                for g, name in viz.PROMPT_TITLES.items():
                    d = oof[oof.prompt_id == g]
                    fig.add_scatter(x=d[tgt], y=d[f"{sel}__{tgt}"], mode="markers", name=name,
                                    marker=dict(size=4.5, opacity=0.4, color=dict(zip(viz.PROMPT_TITLES, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))[g]))
                lo, hi = float(oof[tgt].min()), float(oof[tgt].max())
                fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=viz.INK2, dash="dash"), name="Predicción perfecta")
                fig.update_xaxes(title=f"{tgt} real"); fig.update_yaxes(title=f"{tgt} predicho")
                fig.update_layout(title=f"{mlabel(sel)}: predicho vs. real")
                st.plotly_chart(plot_layout(fig, 400), width="stretch")
            with r:
                st.write("")
                st.write("")
                q = pd.qcut(oof["word_count"], 5, labels=False, duplicates="drop")
                ctr = oof.groupby(q)["word_count"].median().astype(int)
                fig = go.Figure()
                for m in mods:
                    e = [float(np.mean(np.sqrt(np.mean((oof.loc[q == k, T].to_numpy() - oof.loc[q == k, [f"{m}__content", f"{m}__wording"]].to_numpy()) ** 2, axis=0)))) for k in sorted(q.unique())]
                    fig.add_scatter(x=[f"~{c} palabras" for c in ctr], y=e, mode="lines+markers", name=mlabel(m),
                                    line=dict(color=mcolor(m), width=2.5), marker=dict(size=8, line=dict(color=viz.SURFACE, width=2)))
                fig.update_layout(title="Error según la longitud del resumen")
                st.plotly_chart(plot_layout(fig, 400), width="stretch")

            imp_p = RES / "importancia_lgbm.csv"
            if imp_p.exists() and "lgbm_enriquecido" in mods:
                imp = pd.read_csv(imp_p, index_col=0)
                imp["m"] = imp.mean(axis=1)
                top = imp.sort_values("m").tail(12)
                fig = go.Figure()
                for t, col in zip(T, [viz.COLORS["lgbm_enriquecido"], viz.SEQ_BLUE[5]]):
                    fig.add_bar(y=[FEATURE_ES.get(k, k) for k in top.index], x=top[t], orientation="h", name=t,
                                marker=dict(color=col, line=dict(color=viz.SURFACE, width=1)))
                fig.update_layout(barmode="group", title="Qué rasgos usa LightGBM (importancia relativa)")
                st.plotly_chart(plot_layout(fig, 460), width="stretch")
                st.caption("La longitud domina las dos notas. Por eso se compara siempre contra una línea base que solo usa el número de palabras.")
            with st.expander("Tabla completa de resultados (incluye variantes de ablación)"):
                tab = pd.read_csv(RES / "tabla_modelos.csv")
                st.dataframe(tab.round(4), width="stretch", hide_index=True)
