# Proyecto 2 — CommonLit: Evaluar resúmenes de estudiantes · Grupo 6
CC3084 Data Science · UVG · Semestre II 2026 · Reto 12 (Procesamiento del Lenguaje Natural)
Competencia: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview

Se predicen dos notas que un experto asigna al resumen de un estudiante: **content** (cobertura de las
ideas del texto fuente) y **wording** (calidad de la redacción). Métrica: MCRMSE.
La validación es **Leave-One-Prompt-Out**: cada pliegue deja fuera un texto fuente completo, porque la
prueba real usa textos que el modelo no vio.

## Estructura

```
data/            CSV de la competencia (no se versionan; descargar de Kaggle)
notebooks/
  eda_commonlit.ipynb   análisis exploratorio (entrega anterior)
  deberta_gpu.ipynb     ajuste fino de DeBERTa-v3 en GPU (Kaggle/Colab)
src/
  eda_utils.py          limpieza y 21 rasgos del EDA
  features.py           rasgos del modelado (legibilidad, ortografía, TF-IDF/LSA, cobertura)
  train_cpu.py          Ridge y LightGBM con validación anidada Leave-One-Prompt-Out
  train_deberta.py      DeBERTa-v3 con salida doble (lo usa el notebook de GPU)
  evaluate.py           ensamble por stacking, métricas y figuras estáticas
  inference.py          preprocesamiento + predicción para la app
  viz.py                paleta y estilo compartidos
app/app.py              aplicación Streamlit
results/                predicciones fuera de muestra y métricas
models/                 modelos finales entrenados con los 4 textos
figuras/                figuras del EDA y de resultados
informe/                informe final en LaTeX
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
python src/train_cpu.py      # rasgos + Ridge + LightGBM  (~1–2 min en CPU)
# (opcional) correr notebooks/deberta_gpu.ipynb en GPU y copiar results/deberta_* aquí
python src/evaluate.py       # ensamble, métricas y figuras
streamlit run app/app.py     # aplicación
```

## Resultados (MCRMSE medio, un texto fuente fuera en cada pliegue)

| Modelo | MCRMSE |
|---|---|
| Ensamble (Ridge + LightGBM + DeBERTa base) | 0,518 |
| LightGBM, 33 rasgos | 0,524 |
| LightGBM, 18 rasgos | 0,546 |
| Ridge, 33 rasgos | 0,567 |
| Ridge, 18 rasgos | 0,592 |
| DeBERTa-v3 base | 0,608 |
| DeBERTa-v3 small | 0,617 |
| Solo longitud (línea base) | 0,724 |

Detalle en `results/tabla_modelos.csv` y en el informe. Con 4 pliegues, la ventaja del ensamble sobre
LightGBM (0,006) no es concluyente. Los pesos de DeBERTa (`deberta_*_final/`, ~0,5 GB) no se versionan:
colócalos en `results/` (o en `deberta_results/`) para que la app los use.

## Hallazgos que condicionaron el modelado

1. La longitud del resumen es la señal más fuerte (r ≈ 0,86 con content). Toda comparación incluye una
   línea base que solo usa longitud.
2. content y wording están correlacionadas (0,75) pero se predicen con salidas separadas.
3. Copiar y pegar penaliza wording; el solapamiento medio ayuda a content.
4. **Rasgos que identifican el texto fuente se descartan** (`prompt_word_count` y la razón resumen/fuente):
   con solo 4 textos, un modelo lineal los usa para "reconocer" el texto y falla al extrapolar
   (MCRMSE 2,31 en *The Jungle* con ellos, 0,52 sin ellos).
