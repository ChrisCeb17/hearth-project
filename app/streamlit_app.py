"""Demo interactiva del modelo de predicción de enfermedad cardíaca.

Incluye dos modos de uso:
    - Predicción individual: formulario para un paciente a la vez.
    - Procesamiento por lotes: sube un CSV con varios pacientes y descarga las
      predicciones generadas.

Ejecutar con:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load


def find_project_root(marker: str = "models") -> Path:
    """Sube directorios desde este archivo hasta encontrar la carpeta `marker`."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"No se encontró la carpeta '{marker}' en ningún directorio padre.")


PROJECT_ROOT = find_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pipelines.inference_pipeline.inference_pipeline import (  # noqa: E402
    REQUIRED_COLUMNS,
    InferenceDataError,
    generate_predictions,
    prepare_features_for_inference,
)

MODEL_PATH = PROJECT_ROOT / "models" / "corazon_classification-random_forest-v1.joblib"
SAMPLE_INPUT_PATH = PROJECT_ROOT / "app" / "sample_data" / "ejemplo_batch_input.csv"

st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca",
    page_icon="🫀",
    layout="centered",
)


@st.cache_resource
def load_model_cached() -> object:
    """Carga el pipeline entrenado (preprocesamiento + modelo) una sola vez por sesión."""
    return load(MODEL_PATH)


model = load_model_cached()

st.title("🫀 Predicción de Enfermedad Cardíaca")
st.markdown(
    """
Esta demo usa un modelo **Random Forest** entrenado sobre el dataset `corazon.csv`
para estimar la probabilidad de que un paciente tenga enfermedad cardíaca.

⚠️ **Esta herramienta es únicamente demostrativa/educativa** y no reemplaza el
diagnóstico de un profesional de la salud.
"""
)

tab_individual, tab_lote = st.tabs(["🩺 Predicción individual", "📁 Procesamiento por lotes"])

# ====================================================================
# PESTAÑA 1: Predicción individual
# ====================================================================
with tab_individual:
    st.subheader("Datos del paciente")

    with st.form("formulario_paciente"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Edad", min_value=1, max_value=120, value=50, step=1)
            sex = st.selectbox("Sexo", options=["Male", "Female"])
            chest_pain = st.selectbox(
                "Tipo de dolor de pecho",
                options=["typical", "nontypical", "nonanginal", "asymptomatic"],
                help=(
                    "typical: angina típica | nontypical: angina atípica | "
                    "nonanginal: dolor no anginal | asymptomatic: asintomático"
                ),
            )
            rest_bp = st.number_input(
                "Presión arterial en reposo (mm Hg)", min_value=60, max_value=250, value=120
            )
            chol = st.number_input(
                "Colesterol sérico (mg/dl)", min_value=100, max_value=600, value=200
            )
            fbs = st.selectbox(
                "Azúcar en sangre en ayunas > 120 mg/dl",
                options=["No", "Sí"],
                help="Fasting blood sugar (fbs)",
            )
            rest_ecg = st.selectbox(
                "Resultado electrocardiográfico en reposo",
                options=["normal", "left ventricular hypertrophy", "ST-T wave abnormality"],
            )

        with col2:
            max_hr = st.number_input(
                "Frecuencia cardíaca máxima alcanzada", min_value=60, max_value=220, value=150
            )
            exang = st.selectbox(
                "Angina inducida por ejercicio",
                options=["No", "Sí"],
                help="Exercise induced angina (exang)",
            )
            old_peak = st.number_input(
                "Depresión del ST inducida por ejercicio (old_peak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
            slope = st.selectbox(
                "Pendiente del segmento ST en el pico de ejercicio",
                options=[1, 2, 3],
                help="1: ascendente | 2: plana | 3: descendente",
            )
            ca = st.selectbox(
                "Número de vasos principales coloreados por fluoroscopia", options=[0, 1, 2, 3]
            )
            thal = st.selectbox(
                "Talasemia (thal)",
                options=["normal", "fixed", "reversable"],
                help="normal | fixed: defecto fijo | reversable: defecto reversible",
            )

        submitted = st.form_submit_button("🔍 Predecir", use_container_width=True)

    if submitted:
        entrada = pd.DataFrame(
            [
                {
                    "age": float(age),
                    "sex": sex,
                    "chest_pain": chest_pain,
                    "rest_bp": float(rest_bp),
                    "chol": float(chol),
                    "fbs": 1.0 if fbs == "Sí" else 0.0,
                    "rest_ecg": rest_ecg,
                    "max_hr": float(max_hr),
                    "exang": 1.0 if exang == "Sí" else 0.0,
                    "old_peak": float(old_peak),
                    "slope": float(slope),
                    "ca": float(ca),
                    "thal": thal,
                }
            ]
        )

        prediccion = model.predict(entrada)[0]
        probabilidad = model.predict_proba(entrada)[0][1]

        st.divider()
        st.subheader("Resultado")

        if prediccion == 1:
            st.error(
                f"⚠️ **Riesgo de enfermedad cardíaca detectado** (probabilidad: {probabilidad:.1%})"
            )
            st.markdown(
                "El modelo sugiere que este paciente **podría tener enfermedad cardíaca**. "
                "Se recomienda evaluación médica adicional."
            )
        else:
            st.success(f"✅ **Bajo riesgo de enfermedad cardíaca** (probabilidad: {probabilidad:.1%})")
            st.markdown("El modelo sugiere que este paciente **probablemente no tiene enfermedad cardíaca**.")

        st.progress(float(probabilidad))

        with st.expander("Ver datos ingresados"):
            st.dataframe(entrada, use_container_width=True)

        st.caption(
            "Modelo: Random Forest | Métrica de referencia: Recall "
            "(prioriza minimizar falsos negativos en el diagnóstico)."
        )

# ====================================================================
# PESTAÑA 2: Procesamiento por lotes
# ====================================================================
with tab_lote:
    st.subheader("Procesar múltiples pacientes desde un archivo CSV")

    st.markdown(
        f"""
Sube un archivo CSV con uno o varios pacientes. El archivo debe tener las
siguientes columnas (mismos nombres exactos que usa el modelo):

`{", ".join(REQUIRED_COLUMNS)}`

Las filas con datos inválidos (categorías desconocidas, valores no numéricos)
se descartan automáticamente antes de predecir, y se te informa cuántas fueron.
"""
    )

    if SAMPLE_INPUT_PATH.exists():
        with SAMPLE_INPUT_PATH.open("rb") as f:
            st.download_button(
                label="⬇️ Descargar archivo de ejemplo",
                data=f,
                file_name="ejemplo_batch_input.csv",
                mime="text/csv",
                help="Archivo de ejemplo con el formato correcto, para probar la funcionalidad.",
            )

    archivo_subido = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

    if archivo_subido is not None:
        try:
            df_nuevo = pd.read_csv(archivo_subido)
            st.write(f"Archivo leído: **{len(df_nuevo)} fila(s)**, **{len(df_nuevo.columns)} columna(s)**.")

            columnas_faltantes = set(REQUIRED_COLUMNS) - set(df_nuevo.columns)
            if columnas_faltantes:
                st.error(
                    f"❌ Faltan columnas requeridas en el archivo: {sorted(columnas_faltantes)}"
                )
            else:
                with st.spinner("Procesando y generando predicciones..."):
                    df_preparado = prepare_features_for_inference(df_nuevo)
                    df_predicciones = generate_predictions(model, df_preparado)

                n_descartadas = len(df_nuevo) - len(df_preparado)
                if n_descartadas > 0:
                    st.warning(
                        f"⚠️ Se descartaron {n_descartadas} fila(s) con datos inválidos "
                        "antes de predecir."
                    )

                if df_predicciones.empty:
                    st.error("No quedaron filas válidas para predecir.")
                else:
                    n_riesgo = int((df_predicciones["prediction"] == 1).sum())
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Pacientes procesados", len(df_predicciones))
                    col_m2.metric("Con riesgo detectado", n_riesgo)
                    col_m3.metric(
                        "% con riesgo",
                        f"{n_riesgo / len(df_predicciones) * 100:.1f}%" if len(df_predicciones) else "0%",
                    )

                    st.divider()
                    st.subheader("Resultados")
                    st.dataframe(df_predicciones, use_container_width=True)

                    csv_resultado = df_predicciones.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Descargar predicciones (CSV)",
                        data=csv_resultado,
                        file_name="predicciones.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except InferenceDataError as exc:
            st.error(f"❌ Error en los datos: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Ocurrió un error al procesar el archivo: {exc}")
    else:
        st.info("👆 Sube un archivo CSV para comenzar, o descarga el ejemplo de arriba para probar.")

st.divider()
st.caption(
    "Proyecto académico - Ciencia de Datos en Producción | "
    "Dataset: corazon.csv (versión modificada del Heart Disease UCI dataset)"
)
