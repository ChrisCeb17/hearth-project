"""Inference pipeline para el modelo de predicción de enfermedad cardíaca (corazon.csv).

Carga el pipeline entrenado (preprocesamiento + modelo) desde el almacenamiento del
proyecto, lee un lote de datos nuevos desde un archivo, aplica las mismas transformaciones
de limpieza usadas durante el entrenamiento (reutilizando las funciones de
`feature_pipeline.py`), genera las predicciones y las almacena en un archivo.

Uso:
    uv run python src/pipelines/inference_pipeline/inference_pipeline.py
    uv run python src/pipelines/inference_pipeline/inference_pipeline.py \\
        --model models/corazon_classification-random_forest-v1.joblib \\
        --input data/04_inference/nuevos_pacientes.csv \\
        --output data/05_predictions/predicciones.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from pipelines.feature_pipeline.feature_pipeline import (
    clean_boolean_columns,
    clean_ca,
    clean_categorical_columns,
    clean_numeric_columns,
    clean_slope,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
DEFAULT_MODEL_PATH = Path("models/corazon_classification-random_forest-v1.joblib")
DEFAULT_INPUT_PATH = Path("data/04_inference/nuevos_pacientes.csv")
DEFAULT_OUTPUT_PATH = Path("data/05_predictions/predicciones.csv")

# Columnas requeridas en los datos nuevos (mismas features que espera el pipeline entrenado,
# sin la columna objetivo, que no existe en datos de inferencia).
REQUIRED_COLUMNS = [
    "age",
    "sex",
    "chest_pain",
    "rest_bp",
    "chol",
    "fbs",
    "rest_ecg",
    "max_hr",
    "exang",
    "old_peak",
    "slope",
    "ca",
    "thal",
]


class InferenceDataError(Exception):
    """Se lanza cuando los datos de entrada no tienen las columnas requeridas para inferir."""


# ------------------------------------------------------------------
# Carga del modelo
# ------------------------------------------------------------------
def load_model(model_path: Path) -> Pipeline:
    """Carga el pipeline entrenado (preprocesamiento + modelo) desde el almacenamiento.

    Args:
        model_path: ruta al archivo `.joblib` del pipeline entrenado.

    Returns:
        El `Pipeline` de scikit-learn ya ajustado, listo para predecir.

    Raises:
        FileNotFoundError: si el archivo del modelo no existe.
    """
    logger.info("Cargando modelo entrenado desde %s", model_path)
    pipeline = load(model_path)
    logger.info("Modelo cargado: %s", type(pipeline.named_steps["model"]).__name__)
    return pipeline


# ------------------------------------------------------------------
# Carga de datos nuevos
# ------------------------------------------------------------------
def load_new_data(input_path: Path) -> pd.DataFrame:
    """Lee un lote de datos nuevos (pacientes) desde un archivo CSV.

    Args:
        input_path: ruta al archivo CSV con los datos nuevos.

    Returns:
        DataFrame con los datos crudos, tal como llegaron.

    Raises:
        FileNotFoundError: si el archivo no existe.
        InferenceDataError: si faltan columnas requeridas para la inferencia.
    """
    logger.info("Leyendo datos nuevos desde %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Datos nuevos cargados: %s filas, %s columnas", df.shape[0], df.shape[1])

    columnas_faltantes = set(REQUIRED_COLUMNS) - set(df.columns)
    if columnas_faltantes:
        raise InferenceDataError(
            f"Faltan columnas requeridas para la inferencia: {sorted(columnas_faltantes)}"
        )

    return df


# ------------------------------------------------------------------
# Transformaciones (las mismas que en el entrenamiento)
# ------------------------------------------------------------------
def prepare_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las mismas transformaciones de limpieza usadas en el entrenamiento.

    Reutiliza las funciones de limpieza de `feature_pipeline.py` (misma fuente de verdad
    que el entrenamiento), con la excepción de `clean_target`, que no aplica aquí porque
    los datos de inferencia no tienen la columna objetivo.

    Filas que no puedan limpiarse correctamente (categorías inválidas, valores no numéricos)
    se descartan, ya que el modelo no puede predecir de forma confiable sobre datos
    corruptos.

    Args:
        df: DataFrame crudo de datos nuevos (ya validado que tiene las columnas requeridas).

    Returns:
        DataFrame limpio, con el mismo formato de columnas que vio el modelo en
        entrenamiento, listo para pasar a `pipeline.predict()`.
    """
    n_inicial = len(df)

    df = clean_categorical_columns(df)
    df = clean_numeric_columns(df)
    df = clean_slope(df)
    df = clean_ca(df)
    df = clean_boolean_columns(df)

    n_final = len(df)
    if n_final < n_inicial:
        logger.warning(
            "Se descartaron %s de %s fila(s) por datos inválidos antes de predecir",
            n_inicial - n_final,
            n_inicial,
        )

    logger.info("Features preparadas para inferencia: %s filas listas", n_final)
    return df


# ------------------------------------------------------------------
# Generación de predicciones
# ------------------------------------------------------------------
def generate_predictions(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Genera predicciones para cada fila del DataFrame usando el pipeline entrenado.

    Args:
        pipeline: pipeline entrenado (preprocesamiento + modelo).
        df: DataFrame de features ya limpio, listo para predecir.

    Returns:
        Copia de `df` con dos columnas nuevas:
            - "prediction": clase predicha (0 = sin enfermedad, 1 = con enfermedad).
            - "prediction_proba": probabilidad estimada de la clase 1.
    """
    if df.empty:
        logger.warning("No hay filas válidas para predecir")
        resultado = df.copy()
        resultado["prediction"] = pd.Series(dtype="int64")
        resultado["prediction_proba"] = pd.Series(dtype="float64")
        return resultado

    resultado = df.copy()
    resultado["prediction"] = pipeline.predict(df)
    resultado["prediction_proba"] = pipeline.predict_proba(df)[:, 1]

    n_positivos = int((resultado["prediction"] == 1).sum())
    logger.info(
        "Predicciones generadas: %s filas | %s con riesgo detectado (%.1f%%)",
        len(resultado),
        n_positivos,
        n_positivos / len(resultado) * 100 if len(resultado) else 0,
    )
    return resultado


# ------------------------------------------------------------------
# Almacenamiento
# ------------------------------------------------------------------
def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda las predicciones en un archivo CSV.

    Args:
        df: DataFrame con las predicciones (salida de `generate_predictions`).
        output_path: ruta destino del archivo CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Predicciones guardadas en %s (%s filas)", output_path, len(df))


# ------------------------------------------------------------------
# Orquestación / CLI
# ------------------------------------------------------------------
def run_pipeline(model_path: Path, input_path: Path, output_path: Path) -> pd.DataFrame:
    """Ejecuta el inference pipeline completo: carga, transformación, predicción y guardado.

    Args:
        model_path: ruta al pipeline entrenado (`.joblib`).
        input_path: ruta al CSV de datos nuevos.
        output_path: ruta de salida de las predicciones.

    Returns:
        DataFrame con las predicciones generadas (también persistido en disco).
    """
    pipeline = load_model(model_path)
    df_nuevo = load_new_data(input_path)
    df_preparado = prepare_features_for_inference(df_nuevo)
    df_predicciones = generate_predictions(pipeline, df_preparado)
    save_predictions(df_predicciones, output_path)
    return df_predicciones


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Inference pipeline: carga el modelo entrenado, lee datos nuevos, "
        "aplica las mismas transformaciones del entrenamiento y genera predicciones."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Ruta al modelo entrenado (.joblib) (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Ruta al CSV de datos nuevos (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Ruta de salida de las predicciones (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del script."""
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_pipeline(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
