"""Training pipeline para el modelo de predicción de enfermedad cardíaca (corazon.csv).

Lee las features ya procesadas y validadas (salida de `feature_pipeline.py`), separa
train/test, entrena un modelo de clasificación dentro de un pipeline de scikit-learn
(preprocesamiento + modelo), lo evalúa con métricas apropiadas para el problema, y
almacena tanto el pipeline entrenado (`.joblib`) como los resultados de evaluación
(`.json`).

Uso:
    uv run python src/pipelines/training_pipeline/train_pipeline.py
    uv run python src/pipelines/training_pipeline/train_pipeline.py \\
        --input data/02_intermediate/corazon_type_fixed.parquet \\
        --model-output models/corazon_classification-random_forest-v1.joblib \\
        --metrics-output models/metrics/corazon_classification-random_forest-v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend no interactivo, para guardar figuras sin display
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuración del problema
# ------------------------------------------------------------------
TARGET = "disease"

# Columnas numéricas y categóricas nominales, definidas en la etapa de feature engineering.
NUMERIC_COLUMNS = ["age", "rest_bp", "chol", "max_hr", "old_peak", "slope", "ca", "fbs", "exang"]
CATEGORICAL_COLUMNS = ["sex", "chest_pain", "rest_ecg", "thal"]

# Hiperparámetros del modelo final. Ajustar según los resultados de GridSearchCV
# obtenidos en la etapa de selección de modelos (6-model-selection).
MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 7,
    "criterion": "gini",
    "random_state": 42,
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
# Métrica principal del proyecto (ver etapa de modelo base): prioriza minimizar
# Falsos Negativos en el diagnóstico de enfermedad cardíaca.
PRIMARY_METRIC = "recall"

# Umbrales para la validación de la separación train/test.
MIN_TEST_TRAIN_RATIO = 0.05  # el test no debe ser menos del 5% del tamaño del train
MAX_LABEL_PROPORTION_DIFF = 0.15  # diferencia máxima tolerada en proporción de clases
NUMERIC_DRIFT_PVALUE_THRESHOLD = 0.01  # p-valor del test KS por debajo del cual se reporta drift

# Configuración de validación cruzada y análisis de generalización del modelo.
CV_FOLDS = 5
CV_SCORING = ["accuracy", "precision", "recall", "f1"]
# Diferencia máxima tolerada entre score de train y de CV/test antes de sospechar overfitting.
OVERFIT_GAP_THRESHOLD = 0.15
# Score mínimo (train y CV) por debajo del cual se sospecha underfitting.
UNDERFIT_SCORE_THRESHOLD = 0.60

DEFAULT_INPUT_PATH = Path("data/02_intermediate/corazon_type_fixed.parquet")
DEFAULT_MODEL_OUTPUT_PATH = Path("models/corazon_classification-random_forest-v1.joblib")
DEFAULT_METRICS_OUTPUT_PATH = Path("models/metrics/corazon_classification-random_forest-v1.json")


# ------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------
class TrainTestValidationError(Exception):
    """Se lanza cuando se detecta fuga de información (data leakage) entre train y test."""


def load_features(input_path: Path, target: str = TARGET) -> pd.DataFrame:
    """Lee las features procesadas desde un archivo Parquet.

    Args:
        input_path: ruta al archivo Parquet generado por `feature_pipeline.py`.
        target: nombre de la columna objetivo, usada para validar que exista.

    Returns:
        DataFrame con las features listas para entrenar.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si la columna objetivo no está presente en los datos.
    """
    logger.info("Leyendo features procesadas desde %s", input_path)
    df = pd.read_parquet(input_path)

    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no está presente en {input_path}")

    logger.info("Features cargadas: %s filas, %s columnas", df.shape[0], df.shape[1])
    return df


# ------------------------------------------------------------------
# Train / Test split
# ------------------------------------------------------------------
def split_train_test(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa las features en conjuntos de entrenamiento y prueba (estratificado).

    Args:
        df: DataFrame de features completo, incluyendo la columna objetivo.
        target: nombre de la columna objetivo.
        test_size: proporción del conjunto de prueba.
        random_state: semilla para reproducibilidad.

    Returns:
        Tupla `(x_train, x_test, y_train, y_test)`.
    """
    x = df.drop(columns=[target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info("Train: %s filas | Test: %s filas", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


# ------------------------------------------------------------------
# Validación de la separación train/test
# ------------------------------------------------------------------
def _check_index_leakage(x_train: pd.DataFrame, x_test: pd.DataFrame) -> list[str]:
    """Verifica que no haya índices compartidos entre train y test (fuga de datos)."""
    indices_comunes = set(x_train.index) & set(x_test.index)
    if indices_comunes:
        return [f"Fuga de índices: {len(indices_comunes)} índice(s) presentes en train y test"]
    return []


def _check_duplicate_rows_leakage(x_train: pd.DataFrame, x_test: pd.DataFrame) -> list[str]:
    """Verifica que no existan filas idénticas (mismos valores) presentes en ambos conjuntos."""
    filas_en_ambos = pd.merge(
        x_test.reset_index(drop=True), x_train.reset_index(drop=True), how="inner"
    )
    if len(filas_en_ambos) > 0:
        return [
            (
                f"Fuga de datos: {len(filas_en_ambos)} fila(s) de test tienen valores "
                "idénticos a filas de train"
            )
        ]
    return []


def _check_size_ratio(x_train: pd.DataFrame, x_test: pd.DataFrame) -> list[str]:
    """Verifica que el tamaño relativo de test respecto a train sea razonable."""
    if len(x_train) == 0:
        return ["El conjunto de train está vacío"]
    ratio = len(x_test) / len(x_train)
    if ratio < MIN_TEST_TRAIN_RATIO:
        return [
            (
                f"El conjunto de test es muy pequeño respecto a train "
                f"(ratio={ratio:.3f}, mínimo esperado={MIN_TEST_TRAIN_RATIO})"
            )
        ]
    return []


def _check_label_distribution(y_train: pd.Series, y_test: pd.Series) -> list[str]:
    """Verifica que la distribución de clases del target sea similar entre train y test."""
    warnings_list = []
    prop_train = y_train.value_counts(normalize=True)
    prop_test = y_test.value_counts(normalize=True)

    for clase in set(prop_train.index) | set(prop_test.index):
        p_train = prop_train.get(clase, 0.0)
        p_test = prop_test.get(clase, 0.0)
        diff = abs(p_train - p_test)
        if diff > MAX_LABEL_PROPORTION_DIFF:
            warnings_list.append(
                f"Distribución del target difiere para la clase '{clase}': "
                f"train={p_train:.2%}, test={p_test:.2%} (diferencia={diff:.2%})"
            )
    return warnings_list


def _check_new_categories(
    x_train: pd.DataFrame, x_test: pd.DataFrame, categorical_columns: list[str]
) -> list[str]:
    """Verifica si test contiene categorías no vistas en train, para columnas categóricas."""
    warnings_list = []
    for col in categorical_columns:
        if col not in x_train.columns or col not in x_test.columns:
            continue
        categorias_train = set(x_train[col].dropna().unique())
        categorias_test = set(x_test[col].dropna().unique())
        categorias_nuevas = categorias_test - categorias_train
        if categorias_nuevas:
            warnings_list.append(
                f"Columna '{col}': categorías presentes en test pero no en train: "
                f"{sorted(str(c) for c in categorias_nuevas)}"
            )
    return warnings_list


def _check_numeric_drift(
    x_train: pd.DataFrame, x_test: pd.DataFrame, numeric_columns: list[str]
) -> list[str]:
    """Verifica drift entre train y test en columnas numéricas usando el test de Kolmogorov-Smirnov."""
    warnings_list = []
    for col in numeric_columns:
        if col not in x_train.columns or col not in x_test.columns:
            continue
        train_vals = x_train[col].dropna()
        test_vals = x_test[col].dropna()
        min_samples_ks_test = 2
        if len(train_vals) < min_samples_ks_test or len(test_vals) < min_samples_ks_test:
            continue

        statistic, p_value = stats.ks_2samp(train_vals, test_vals)
        if p_value < NUMERIC_DRIFT_PVALUE_THRESHOLD:
            warnings_list.append(
                f"Columna '{col}': posible drift entre train y test "
                f"(KS statistic={statistic:.3f}, p-value={p_value:.4f})"
            )
    return warnings_list


def validate_train_test_split(  # noqa: PLR0913, PLR0917
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_columns: list[str] = NUMERIC_COLUMNS,
    categorical_columns: list[str] = CATEGORICAL_COLUMNS,
) -> dict[str, Any]:
    """Valida la separación train/test: evita fuga de información y verifica representatividad.

    Ejecuta los siguientes checks:
        - Fuga de índices entre train y test (crítico -> error).
        - Fuga por filas duplicadas con valores idénticos entre train y test (crítico -> error).
        - Tamaño relativo de test respecto a train (advertencia).
        - Distribución del target (label) similar entre train y test (advertencia).
        - Categorías nuevas en test no vistas en train (advertencia).
        - Drift de variables numéricas entre train y test, vía test de Kolmogorov-Smirnov
          (advertencia).

    Args:
        x_train: features de entrenamiento.
        x_test: features de prueba.
        y_train: target de entrenamiento.
        y_test: target de prueba.
        numeric_columns: columnas numéricas a evaluar por drift.
        categorical_columns: columnas categóricas a evaluar por categorías nuevas.

    Returns:
        Diccionario con:
            - "passed": True si no hubo fuga de datos (independiente de las advertencias).
            - "leakage_detected": True si se detectó fuga de datos (índices o filas duplicadas).
            - "errors": lista de mensajes de error crítico (fuga de datos).
            - "warnings": lista de mensajes de advertencia (no crítico).

    Raises:
        TrainTestValidationError: si se detecta fuga de información (data leakage) entre
            train y test. En ese caso, el pipeline NO debe continuar con el entrenamiento.
    """
    errores = _check_index_leakage(x_train, x_test) + _check_duplicate_rows_leakage(x_train, x_test)

    advertencias = (
        _check_size_ratio(x_train, x_test)
        + _check_label_distribution(y_train, y_test)
        + _check_new_categories(x_train, x_test, categorical_columns)
        + _check_numeric_drift(x_train, x_test, numeric_columns)
    )

    for advertencia in advertencias:
        logger.warning("Validación train/test: %s", advertencia)

    if errores:
        for error in errores:
            logger.error("Validación train/test: %s", error)
        mensaje = (
            f"Se detectó fuga de información (data leakage) entre train y test: "
            f"{'; '.join(errores)}"
        )
        raise TrainTestValidationError(mensaje)

    if not advertencias:
        logger.info("Validación train/test EXITOSA: sin fugas de datos ni advertencias")
    else:
        logger.info(
            "Validación train/test completada sin fuga de datos, con %s advertencia(s)",
            len(advertencias),
        )

    return {
        "passed": True,
        "leakage_detected": False,
        "errors": errores,
        "warnings": advertencias,
    }


# ------------------------------------------------------------------
# Pipeline de preprocesamiento + modelo
# ------------------------------------------------------------------
def build_preprocessor(
    numeric_columns: list[str] = NUMERIC_COLUMNS,
    categorical_columns: list[str] = CATEGORICAL_COLUMNS,
) -> ColumnTransformer:
    """Construye el ColumnTransformer de preprocesamiento (mismo criterio de feat_eng).

    Args:
        numeric_columns: columnas numéricas a imputar y escalar.
        categorical_columns: columnas categóricas a imputar y codificar (One-Hot).

    Returns:
        `ColumnTransformer` sin ajustar.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_columns),
            ("categoric", categorical_pipe, categorical_columns),
        ]
    )


def build_model(model_params: dict[str, Any] = MODEL_PARAMS) -> RandomForestClassifier:
    """Construye el clasificador con los hiperparámetros del modelo seleccionado.

    Args:
        model_params: hiperparámetros del `RandomForestClassifier`.

    Returns:
        Instancia del clasificador (sin ajustar).
    """
    return RandomForestClassifier(**model_params)


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer | None = None,
    model_params: dict[str, Any] = MODEL_PARAMS,
) -> Pipeline:
    """Entrena el pipeline completo (preprocesamiento + modelo) sobre el set de train.

    Args:
        x_train: features de entrenamiento.
        y_train: target de entrenamiento.
        preprocessor: `ColumnTransformer` a usar. Si es `None`, se construye uno nuevo.
        model_params: hiperparámetros del modelo.

    Returns:
        `Pipeline` de scikit-learn ya ajustado.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    model = build_model(model_params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    logger.info("Entrenando modelo %s con parámetros: %s", type(model).__name__, model_params)
    pipeline.fit(x_train, y_train)
    logger.info("Entrenamiento finalizado")
    return pipeline


# ------------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------------
def evaluate_model(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evalúa el pipeline entrenado sobre el set de prueba con métricas de clasificación.

    Args:
        pipeline: pipeline entrenado (preprocesamiento + modelo).
        x_test: features de prueba.
        y_test: target de prueba.

    Returns:
        Diccionario con métricas: accuracy, precision, recall, f1, roc_auc y matriz
        de confusión (como lista anidada, para ser serializable a JSON).
    """
    y_pred = pipeline.predict(x_test)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test_samples": len(y_test),
    }

    # roc_auc requiere probabilidades; solo se calcula si el modelo las soporta.
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    logger.info(
        "Evaluación en test | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


# ------------------------------------------------------------------
# Validación robusta del modelo: cross-validation, comparación y diagnóstico
# ------------------------------------------------------------------
def get_cv_splitter(n_splits: int = CV_FOLDS) -> StratifiedKFold:
    """Devuelve el splitter de validación cruzada apropiado para este problema.

    Se usa `StratifiedKFold` (no `KFold` ni `TimeSeriesSplit`) porque:
        - Es un problema de clasificación binaria con clases moderadamente balanceadas
          (no perfectamente 50/50), por lo que estratificar mantiene la proporción de
          clases en cada fold, evitando folds con distribuciones sesgadas.
        - No hay una dimensión temporal en los datos (no es una serie de tiempo), por lo
          que `TimeSeriesSplit` no aplica.

    Args:
        n_splits: número de particiones de la validación cruzada.

    Returns:
        Instancia de `StratifiedKFold` configurada con semilla fija para reproducibilidad.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict[str, Any] = MODEL_PARAMS,
    cv_folds: int = CV_FOLDS,
    scoring: list[str] = CV_SCORING,
) -> dict[str, Any]:
    """Ejecuta validación cruzada sobre el set de entrenamiento con múltiples métricas.

    Construye un pipeline SIN ajustar (mismo preprocesador + mismos hiperparámetros que
    `train_model`) y lo evalúa con `cross_validate`, que ajusta y evalúa un modelo nuevo
    en cada fold — evitando cualquier fuga entre folds.

    Args:
        x_train: features de entrenamiento (no se usa test, para no filtrar información).
        y_train: target de entrenamiento.
        model_params: hiperparámetros del modelo a validar.
        cv_folds: número de folds de la validación cruzada.
        scoring: lista de métricas a calcular en cada fold.

    Returns:
        Diccionario `{métrica: {"mean": float, "std": float, "scores": list[float]}}`
        con los resultados de cada métrica a través de los folds.
    """
    preprocessor = build_preprocessor()
    model = build_model(model_params)
    pipeline_sin_ajustar = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    splitter = get_cv_splitter(n_splits=cv_folds)

    logger.info("Ejecutando validación cruzada (%s-fold, StratifiedKFold)", cv_folds)
    resultados_cv = cross_validate(
        pipeline_sin_ajustar,
        x_train,
        y_train,
        cv=splitter,
        scoring=scoring,
        n_jobs=-1,
    )

    resumen: dict[str, Any] = {}
    for metrica in scoring:
        scores = resultados_cv[f"test_{metrica}"]
        resumen[metrica] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": [float(s) for s in scores],
        }
        logger.info(
            "CV %s | media=%.4f desviación estándar=%.4f",
            metrica,
            resumen[metrica]["mean"],
            resumen[metrica]["std"],
        )

    return resumen


def compare_train_cv_test(
    train_metrics: dict[str, Any],
    cv_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    metrics_to_compare: list[str] = CV_SCORING,
) -> dict[str, Any]:
    """Construye una tabla comparativa de métricas entre train, validación cruzada y test.

    Args:
        train_metrics: salida de `evaluate_model` sobre el set de entrenamiento.
        cv_metrics: salida de `cross_validate_model`.
        test_metrics: salida de `evaluate_model` sobre el set de prueba.
        metrics_to_compare: métricas a incluir en la comparación.

    Returns:
        Diccionario `{métrica: {"train": float, "cv_mean": float, "cv_std": float,
        "test": float}}`.
    """
    comparacion: dict[str, Any] = {}
    for metrica in metrics_to_compare:
        comparacion[metrica] = {
            "train": train_metrics[metrica],
            "cv_mean": cv_metrics[metrica]["mean"],
            "cv_std": cv_metrics[metrica]["std"],
            "test": test_metrics[metrica],
        }
        logger.info(
            "Comparación %s | train=%.4f | cv=%.4f (+/-%.4f) | test=%.4f",
            metrica,
            comparacion[metrica]["train"],
            comparacion[metrica]["cv_mean"],
            comparacion[metrica]["cv_std"],
            comparacion[metrica]["test"],
        )
    return comparacion


def analyze_generalization(
    comparison: dict[str, Any],
    primary_metric: str = PRIMARY_METRIC,
    overfit_gap_threshold: float = OVERFIT_GAP_THRESHOLD,
    underfit_score_threshold: float = UNDERFIT_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Diagnostica underfitting/overfitting comparando train, CV y test en la métrica principal.

    Reglas de diagnóstico (sobre `primary_metric`, por defecto Recall):
        - **Underfitting**: el score de train Y el de CV están ambos por debajo de
          `underfit_score_threshold` — el modelo ni siquiera ajusta bien los datos que ve.
        - **Overfitting**: la brecha entre train y CV, o entre train y test, supera
          `overfit_gap_threshold` — el modelo memoriza train pero no generaliza.
        - **Buena generalización**: ninguna de las condiciones anteriores se cumple.

    Args:
        comparison: salida de `compare_train_cv_test`.
        primary_metric: métrica usada para el diagnóstico (debe existir en `comparison`).
        overfit_gap_threshold: brecha máxima tolerada antes de sospechar overfitting.
        underfit_score_threshold: score mínimo esperado antes de sospechar underfitting.

    Returns:
        Diccionario con `status` ("overfitting", "underfitting" o "buena_generalizacion"),
        las brechas calculadas, y una lista de `recommendations` (acciones sugeridas).
    """
    valores = comparison[primary_metric]
    train_score = valores["train"]
    cv_score = valores["cv_mean"]
    test_score = valores["test"]

    gap_train_cv = train_score - cv_score
    gap_train_test = train_score - test_score

    recomendaciones: list[str] = []

    if train_score < underfit_score_threshold and cv_score < underfit_score_threshold:
        status = "underfitting"
        recomendaciones = [
            (
                "El modelo no logra un buen desempeño ni siquiera en entrenamiento: "
                "considerar un modelo más complejo (más árboles, mayor profundidad)."
            ),
            (
                "Revisar si faltan features relevantes o si el feature engineering "
                "actual es insuficiente."
            ),
            (
                "Verificar que las variables más predictivas no se hayan perdido en la "
                "limpieza de datos."
            ),
        ]
    elif gap_train_cv > overfit_gap_threshold or gap_train_test > overfit_gap_threshold:
        status = "overfitting"
        recomendaciones = [
            (
                "El modelo generaliza peor de lo esperado: considerar reducir la "
                "complejidad (menor max_depth, más regularización)."
            ),
            "Evaluar conseguir más datos de entrenamiento, si es posible.",
            (
                "Revisar si el conjunto de train tiene distribución muy distinta a "
                "CV/test (ver validate_train_test_split)."
            ),
        ]
    else:
        status = "buena_generalizacion"
        recomendaciones = [
            (
                "El modelo generaliza razonablemente bien entre train, validación "
                "cruzada y test. Se puede proceder con este modelo como referencia."
            )
        ]

    logger.info(
        "Diagnóstico de generalización (%s): %s | gap train-cv=%.4f | gap train-test=%.4f",
        primary_metric,
        status,
        gap_train_cv,
        gap_train_test,
    )
    for r in recomendaciones:
        logger.info("Recomendación: %s", r)

    return {
        "status": status,
        "primary_metric": primary_metric,
        "train_score": train_score,
        "cv_mean_score": cv_score,
        "test_score": test_score,
        "gap_train_cv": gap_train_cv,
        "gap_train_test": gap_train_test,
        "recommendations": recomendaciones,
    }


def plot_cv_scores(cv_metrics: dict[str, Any], output_path: Path) -> None:
    """Genera y guarda un boxplot con la distribución de scores de la validación cruzada.

    Args:
        cv_metrics: salida de `cross_validate_model`.
        output_path: ruta donde guardar la figura (PNG).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datos = {metrica: valores["scores"] for metrica, valores in cv_metrics.items()}
    df_scores = pd.DataFrame(datos)

    fig, ax = plt.subplots(figsize=(8, 5))
    df_scores.plot.box(ax=ax)
    ax.set_title(f"Distribución de métricas en validación cruzada ({CV_FOLDS}-fold)")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info("Gráfica de validación cruzada guardada en %s", output_path)


# ------------------------------------------------------------------
# Almacenamiento
# ------------------------------------------------------------------
def save_model(pipeline: Pipeline, output_path: Path) -> None:
    """Guarda el pipeline entrenado (preprocesamiento + modelo) en formato joblib.

    Args:
        pipeline: pipeline entrenado.
        output_path: ruta destino del archivo `.joblib`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, output_path)
    logger.info("Modelo guardado en %s", output_path)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Guarda las métricas de evaluación en formato JSON.

    Args:
        metrics: diccionario de métricas (ver `evaluate_model`).
        output_path: ruta destino del archivo `.json`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Métricas guardadas en %s", output_path)


# ------------------------------------------------------------------
# Orquestación / CLI
# ------------------------------------------------------------------
def run_pipeline(  # noqa: PLR0913, PLR0917
    input_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    model_params: dict[str, Any] = MODEL_PARAMS,
) -> dict[str, Any]:
    """Ejecuta el training pipeline completo: lectura, split, entrenamiento, evaluación y guardado.

    Args:
        input_path: ruta del Parquet de features procesadas.
        model_output_path: ruta de salida del pipeline entrenado (`.joblib`).
        metrics_output_path: ruta de salida de las métricas de evaluación (`.json`).
        test_size: proporción del conjunto de prueba.
        random_state: semilla para reproducibilidad.
        model_params: hiperparámetros del modelo.

    Returns:
        Diccionario con las métricas de test (en el nivel superior, para compatibilidad),
        más las claves "train_metrics", "cross_validation", "train_cv_test_comparison" y
        "generalization_analysis" con el detalle completo de la validación del modelo.
    """
    df = load_features(input_path)
    x_train, x_test, y_train, y_test = split_train_test(
        df, test_size=test_size, random_state=random_state
    )

    validate_train_test_split(x_train, x_test, y_train, y_test)

    pipeline = train_model(x_train, y_train, model_params=model_params)

    # Evaluación en los tres frentes: train (ajuste), CV (estabilidad), test (generalización)
    train_metrics = evaluate_model(pipeline, x_train, y_train)
    cv_metrics = cross_validate_model(x_train, y_train, model_params=model_params)
    metrics = evaluate_model(pipeline, x_test, y_test)

    comparison = compare_train_cv_test(train_metrics, cv_metrics, metrics)
    diagnosis = analyze_generalization(comparison)

    save_model(pipeline, model_output_path)

    reporte_completo = {
        **metrics,
        "train_metrics": train_metrics,
        "cross_validation": cv_metrics,
        "train_cv_test_comparison": comparison,
        "generalization_analysis": diagnosis,
    }
    save_metrics(reporte_completo, metrics_output_path)

    cv_plot_path = metrics_output_path.with_name(metrics_output_path.stem + "_cv_boxplot.png")
    plot_cv_scores(cv_metrics, cv_plot_path)

    return reporte_completo


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Training pipeline: lee features procesadas, entrena y evalúa el modelo, "
        "y guarda el pipeline entrenado junto con las métricas de evaluación."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Ruta al Parquet de features procesadas (default: {DEFAULT_INPUT_PATH})",
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
        help=f"Ruta de salida de las métricas (default: {DEFAULT_METRICS_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Proporción del conjunto de prueba (default: {TEST_SIZE})",
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
    try:
        run_pipeline(
            input_path=args.input,
            model_output_path=args.model_output,
            metrics_output_path=args.metrics_output,
            test_size=args.test_size,
        )
    except TrainTestValidationError:
        logger.exception("Pipeline detenido")
        sys.exit(1)


if __name__ == "__main__":
    main()
