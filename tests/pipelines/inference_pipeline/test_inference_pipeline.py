"""Pruebas unitarias para inference_pipeline.py, usando un modelo dummy y datos sintéticos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipelines.inference_pipeline.inference_pipeline import (
    InferenceDataError,
    generate_predictions,
    load_model,
    load_new_data,
    prepare_features_for_inference,
    run_pipeline,
    save_predictions,
)

NUMERIC_COLUMNS = ["age", "rest_bp", "chol", "max_hr", "old_peak", "slope", "ca", "fbs", "exang"]
CATEGORICAL_COLUMNS = ["sex", "chest_pain", "rest_ecg", "thal"]


@pytest.fixture
def synthetic_training_data() -> pd.DataFrame:
    """Dataset sintético pequeño, con el mismo esquema de columnas que corazon.csv limpio."""
    rng = np.random.default_rng(42)
    n = 40
    age = rng.integers(30, 75, n).astype(float)
    max_hr = np.minimum(220 - age, rng.integers(100, 180, n)).astype(float)

    return pd.DataFrame(
        {
            "age": age,
            "sex": rng.choice(["Male", "Female"], n),
            "chest_pain": rng.choice(["typical", "nontypical", "nonanginal", "asymptomatic"], n),
            "rest_bp": rng.integers(100, 180, n).astype(float),
            "chol": rng.integers(150, 350, n).astype(float),
            "fbs": rng.choice([0.0, 1.0], n),
            "rest_ecg": rng.choice(
                ["normal", "left ventricular hypertrophy", "ST-T wave abnormality"], n
            ),
            "max_hr": max_hr,
            "exang": rng.choice([0.0, 1.0], n),
            "old_peak": np.round(rng.uniform(0, 4, n), 1),
            "slope": rng.choice([1.0, 2.0, 3.0], n),
            "ca": rng.choice([0.0, 1.0, 2.0, 3.0], n),
            "thal": rng.choice(["normal", "fixed", "reversable"], n),
            "disease": rng.choice([0, 1], n),
        }
    )


@pytest.fixture
def dummy_model_path(tmp_path: Path, synthetic_training_data: pd.DataFrame) -> Path:
    """Entrena un pipeline dummy (DummyClassifier) sobre datos sintéticos y lo guarda en disco."""
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_COLUMNS),
            ("categoric", categorical_pipe, CATEGORICAL_COLUMNS),
        ]
    )
    modelo_dummy = DummyClassifier(strategy="stratified", random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", modelo_dummy)])

    x = synthetic_training_data.drop(columns=["disease"])
    y = synthetic_training_data["disease"]
    pipeline.fit(x, y)

    model_path = tmp_path / "modelo_dummy.joblib"
    dump(pipeline, model_path)
    return model_path


@pytest.fixture
def datos_nuevos_validos(synthetic_training_data: pd.DataFrame) -> pd.DataFrame:
    """Datos "nuevos" de inferencia: mismo esquema, sin la columna objetivo."""
    return synthetic_training_data.drop(columns=["disease"]).head(10).reset_index(drop=True)


# ------------------------------------------------------------------
# Carga del modelo
# ------------------------------------------------------------------
class TestLoadModel:
    def test_carga_modelo_correctamente(self, dummy_model_path: Path) -> None:
        pipeline = load_model(dummy_model_path)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_modelo_es_utilizable_tras_cargar(
        self, dummy_model_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        pipeline = load_model(dummy_model_path)
        predicciones = pipeline.predict(datos_nuevos_validos)
        assert len(predicciones) == len(datos_nuevos_validos)

    def test_archivo_inexistente_lanza_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "no_existe.joblib")


# ------------------------------------------------------------------
# Carga de datos nuevos
# ------------------------------------------------------------------
class TestLoadNewData:
    def test_lee_csv_correctamente(
        self, tmp_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        csv_path = tmp_path / "nuevos.csv"
        datos_nuevos_validos.to_csv(csv_path, index=False)

        resultado = load_new_data(csv_path)

        assert len(resultado) == len(datos_nuevos_validos)

    def test_archivo_inexistente_lanza_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_new_data(tmp_path / "no_existe.csv")

    def test_columnas_faltantes_lanza_error(self, tmp_path: Path) -> None:
        df_incompleto = pd.DataFrame({"age": [50, 60], "sex": ["Male", "Female"]})
        csv_path = tmp_path / "incompleto.csv"
        df_incompleto.to_csv(csv_path, index=False)

        with pytest.raises(InferenceDataError):
            load_new_data(csv_path)


# ------------------------------------------------------------------
# Transformaciones (mismas que en entrenamiento)
# ------------------------------------------------------------------
class TestPrepareFeaturesForInference:
    def test_datos_validos_se_conservan(self, datos_nuevos_validos: pd.DataFrame) -> None:
        resultado = prepare_features_for_inference(datos_nuevos_validos)
        assert len(resultado) == len(datos_nuevos_validos)

    def test_convierte_columnas_categoricas_a_category(
        self, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        resultado = prepare_features_for_inference(datos_nuevos_validos)
        assert resultado["sex"].dtype.name == "category"

    def test_elimina_categoria_invalida(self, datos_nuevos_validos: pd.DataFrame) -> None:
        df = datos_nuevos_validos.copy()
        df.loc[0, "thal"] = "categoria_invalida_xyz"

        resultado = prepare_features_for_inference(df)

        assert len(resultado) == len(df) - 1
        assert "categoria_invalida_xyz" not in resultado["thal"].astype(str).values

    def test_elimina_valor_no_numerico(self, datos_nuevos_validos: pd.DataFrame) -> None:
        df = datos_nuevos_validos.copy()
        df["age"] = df["age"].astype(object)
        df.loc[0, "age"] = "no_es_numero"

        resultado = prepare_features_for_inference(df)

        assert len(resultado) == len(df) - 1


# ------------------------------------------------------------------
# Generación de predicciones
# ------------------------------------------------------------------
class TestGeneratePredictions:
    def test_agrega_columnas_de_prediccion(
        self, dummy_model_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        pipeline = load_model(dummy_model_path)
        resultado = generate_predictions(pipeline, datos_nuevos_validos)

        assert "prediction" in resultado.columns
        assert "prediction_proba" in resultado.columns
        assert len(resultado) == len(datos_nuevos_validos)

    def test_predicciones_son_binarias(
        self, dummy_model_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        pipeline = load_model(dummy_model_path)
        resultado = generate_predictions(pipeline, datos_nuevos_validos)

        assert set(resultado["prediction"].unique()).issubset({0, 1})

    def test_probabilidades_en_rango_valido(
        self, dummy_model_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        pipeline = load_model(dummy_model_path)
        resultado = generate_predictions(pipeline, datos_nuevos_validos)

        assert resultado["prediction_proba"].between(0, 1).all()

    def test_dataframe_vacio_no_falla(self, dummy_model_path: Path) -> None:
        pipeline = load_model(dummy_model_path)
        df_vacio = pd.DataFrame(columns=NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)

        resultado = generate_predictions(pipeline, df_vacio)

        assert len(resultado) == 0
        assert "prediction" in resultado.columns
        assert "prediction_proba" in resultado.columns

    def test_conserva_columnas_originales(
        self, dummy_model_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        pipeline = load_model(dummy_model_path)
        resultado = generate_predictions(pipeline, datos_nuevos_validos)

        for col in datos_nuevos_validos.columns:
            assert col in resultado.columns


# ------------------------------------------------------------------
# Almacenamiento
# ------------------------------------------------------------------
class TestSavePredictions:
    def test_guarda_csv_correctamente(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"prediction": [0, 1], "prediction_proba": [0.2, 0.8]})
        output_path = tmp_path / "subdir" / "predicciones.csv"

        save_predictions(df, output_path)

        assert output_path.exists()
        resultado = pd.read_csv(output_path)
        assert len(resultado) == 2  # noqa: PLR2004

    def test_crea_directorios_faltantes(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"prediction": [1]})
        output_path = tmp_path / "no_existe" / "aun" / "predicciones.csv"

        save_predictions(df, output_path)

        assert output_path.parent.exists()


# ------------------------------------------------------------------
# Integración: pipeline completo de punta a punta
# ------------------------------------------------------------------
class TestRunPipeline:
    def test_genera_predicciones_end_to_end(
        self, tmp_path: Path, dummy_model_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        input_path = tmp_path / "nuevos.csv"
        output_path = tmp_path / "predicciones.csv"
        datos_nuevos_validos.to_csv(input_path, index=False)

        resultado = run_pipeline(dummy_model_path, input_path, output_path)

        assert output_path.exists()
        assert "prediction" in resultado.columns
        assert len(resultado) == len(datos_nuevos_validos)

    def test_archivo_de_salida_es_legible(
        self, tmp_path: Path, dummy_model_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        input_path = tmp_path / "nuevos.csv"
        output_path = tmp_path / "predicciones.csv"
        datos_nuevos_validos.to_csv(input_path, index=False)

        run_pipeline(dummy_model_path, input_path, output_path)

        predicciones_guardadas = pd.read_csv(output_path)
        assert "prediction" in predicciones_guardadas.columns
        assert "prediction_proba" in predicciones_guardadas.columns

    def test_datos_con_ruido_se_limpian_antes_de_predecir(
        self, tmp_path: Path, dummy_model_path: Path, datos_nuevos_validos: pd.DataFrame
    ) -> None:
        df_con_ruido = datos_nuevos_validos.copy()
        df_con_ruido.loc[0, "thal"] = "categoria_invalida"

        input_path = tmp_path / "nuevos_con_ruido.csv"
        output_path = tmp_path / "predicciones.csv"
        df_con_ruido.to_csv(input_path, index=False)

        resultado = run_pipeline(dummy_model_path, input_path, output_path)

        assert len(resultado) == len(df_con_ruido) - 1
