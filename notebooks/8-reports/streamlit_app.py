"""Demo interactiva del modelo de predicción de enfermedad cardíaca.

Ejecutar con:
    uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

# ------------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca",
    page_icon="🫀",
    layout="centered",
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "corazon_classification-random_forest-v1.joblib"
)


@st.cache_resource
def load_model():
    """Carga el pipeline entrenado (preprocesamiento + modelo) una sola vez."""
    return load(MODEL_PATH)


model = load_model()

st.title("🫀 Predicción de Enfermedad Cardíaca")
st.markdown(
    """
Esta demo usa un modelo **Random Forest** entrenado sobre el dataset `corazon.csv`
para estimar la probabilidad de que un paciente tenga enfermedad cardíaca, a partir
de datos clínicos básicos.

⚠️ **Esta herramienta es únicamente demostrativa/educativa** y no reemplaza el
diagnóstico de un profesional de la salud.
"""
)

st.divider()

# ------------------------------------------------------------------
# Formulario de entrada
# ------------------------------------------------------------------
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
            "Frecuencia cardíaca máxima alcanzada",
            min_value=60,
            max_value=220,
            value=150,
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
            "Número de vasos principales coloreados por fluoroscopia",
            options=[0, 1, 2, 3],
        )
        thal = st.selectbox(
            "Talasemia (thal)",
            options=["normal", "fixed", "reversable"],
            help="normal | fixed: defecto fijo | reversable: defecto reversible",
        )

    submitted = st.form_submit_button("🔍 Predecir", use_container_width=True)

# ------------------------------------------------------------------
# Predicción
# ------------------------------------------------------------------
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
        st.success(
            f"✅ **Bajo riesgo de enfermedad cardíaca** (probabilidad: {probabilidad:.1%})"
        )
        st.markdown(
            "El modelo sugiere que este paciente **probablemente no tiene enfermedad cardíaca**."
        )

    st.progress(float(probabilidad))

    with st.expander("Ver datos ingresados"):
        st.dataframe(entrada, use_container_width=True)

    st.caption(
        "Modelo: Random Forest | Métrica de referencia: Recall "
        "(prioriza minimizar falsos negativos en el diagnóstico)."
    )

st.divider()
st.caption(
    "Proyecto académico - Ciencia de Datos en Producción | "
    "Dataset: corazon.csv (versión modificada del Heart Disease UCI dataset)"
)
