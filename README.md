# Proyecto 2 — Análisis Exploratorio
### Reto 12: CommonLit — Evaluar resúmenes de estudiantes · Grupo 6

Tema: Procesamiento del Lenguaje Natural.
Competencia: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview

Se predicen dos notas que un experto asigna al resumen de un estudiante:
**content** (cobertura de las ideas del texto fuente) y **wording** (calidad de
la redacción). Este repositorio contiene el análisis exploratorio.

## Cómo ejecutar

```bash
pip install -r requirements.txt
python src/run_eda.py            # genera figuras/ y data/features_train.parquet
```

## Datos en números

- 7165 resúmenes en entrenamiento, **solo 4 textos fuente**.
- Sin valores faltantes, sin duplicados, resumen más corto = 22 palabras.
- El `test.csv` visible es solo un formato de 4 filas; la prueba real usa textos
  nuevos.

## Hallazgos clave

1. **La longitud del resumen es la señal más fuerte** (cantidad de caracteres:
   correlación ~0.86 con content, ~0.61 con wording). Riesgo: un modelo ingenuo
   podría reducirse a contar palabras.
2. **content y wording están correlacionadas** (Pearson 0.75) pero miden cosas
   distintas → predecir juntas, no unir.
3. **Copiar y pegar penaliza**: el solapamiento alto con el texto fuente baja
   ambas notas, sobre todo wording.
4. **La diversidad léxica** correlaciona en negativo, pero es un efecto del
   tamaño → hay que ajustarla.
5. **Hay sesgo por texto fuente** en wording (The Third Wave +0.52 vs The Jungle
   −0.30) → no usar el id del texto como variable y validar por grupo.
6. Los outliers de tamaño (~6%) son casos reales (transcripciones o respuestas
   mínimas), no errores.

## Decisión de validación

**GroupKFold por `prompt_id`** (4 grupos) o Leave-One-Prompt-Out, reportando el
error por texto para vigilar el sobreajuste.
