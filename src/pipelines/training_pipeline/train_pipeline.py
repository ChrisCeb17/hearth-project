"""Training pipeline para el modelo de predicción de enfermedad cardíaca.

Lee las features procesadas desde un archivo Parquet, separa train/test,
entrena un clasificador, evalúa métricas de clasificación y persiste
el modelo junto con los resultados de evaluación.

Uso:
    uv run python src/pipelines/training_pipeline/train_pipeline.py
    uv run python src/pipelines/training_pipeline/train_pipeline.py \
        --input data/02_intermediate/corazon_type_fixed.parquet \
        --model-output data/06_models/corazon_model.joblib \
        --metrics-output data/08_reporting/training_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

TARGET = "disease"
DEFAULT_INPUT_PATH = Path("data/02_intermediate/corazon_type_fixed.parquet")
DEFAULT_MODEL_OUTPUT_PATH = Path("data/06_models/corazon_model.joblib")
DEFAULT_METRICS_OUTPUT_PATH = Path("data/08_reporting/training_metrics.json")


def load_features_data(input_path: Path) -> pd.DataFrame:
    """Lee las features procesadas desde un archivo Parquet."""
    logger.info("Leyendo features desde %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Features cargadas: %s filas, %s columnas", df.shape[0], df.shape[1])
    return df


def split_features_target(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y target."""
    if target not in df.columns:
        msg = f"La columna objetivo '{target}' no existe en el dataset"
        raise ValueError(msg)

    x = df.drop(columns=[target])
    y = df[target].astype(int)
    return x, y


def build_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Entrena un pipeline de clasificación con one-hot encoding + random forest."""
    categorical_columns = x_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns
    numeric_columns = x_train.select_dtypes(
        exclude=["object", "category", "bool", "string"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_columns)),
            ("num", "passthrough", list(numeric_columns)),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evalúa el modelo y retorna métricas de clasificación."""
    y_pred = model.predict(x_test)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    y_prob = model.predict_proba(x_test)[:, 1]
    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    return metrics


def save_model(model: Pipeline, model_output_path: Path) -> None:
    """Guarda el modelo entrenado en disco."""
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    logger.info("Modelo guardado en %s", model_output_path)


def save_metrics(metrics: dict[str, float], metrics_output_path: Path) -> None:
    """Guarda las métricas de evaluación en formato JSON."""
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    logger.info("Métricas guardadas en %s", metrics_output_path)


def run_pipeline(
    input_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Ejecuta entrenamiento completo: lectura, split, entrenamiento, evaluación y guardado."""
    df = load_features_data(input_path)
    x, y = split_features_target(df)

    stratify = y if y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = build_model(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    save_model(model, model_output_path)
    save_metrics(metrics, metrics_output_path)

    return metrics


def parse_args() -> argparse.Namespace:
    """Define y parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Training pipeline: lee features procesadas, entrena un modelo de clasificación "
            "de enfermedad cardíaca, evalúa métricas y guarda artefactos."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Ruta al parquet de features (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT_PATH,
        help=f"Ruta de salida del modelo entrenado (default: {DEFAULT_MODEL_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT_PATH,
        help=f"Ruta de salida de métricas (default: {DEFAULT_METRICS_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción del set de prueba (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)",
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

    metrics = run_pipeline(
        input_path=args.input,
        model_output_path=args.model_output,
        metrics_output_path=args.metrics_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    logger.info("Entrenamiento finalizado. Métricas: %s", metrics)


if __name__ == "__main__":
    main()
