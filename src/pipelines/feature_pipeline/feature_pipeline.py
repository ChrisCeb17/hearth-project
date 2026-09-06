"""Feature pipeline para el proyecto de predicción de enfermedad cardíaca (corazon.csv).

Lee los datos crudos, limpia tipos y valores inválidos, y almacena las features
resultantes en un archivo Parquet, listas para ser consumidas por el pipeline de
entrenamiento (feature engineering con scikit-learn).

Uso:
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py \\
        --input data/01_raw/corazon.csv \\
        --output data/02_intermediate/corazon_type_fixed.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuración de columnas y valores válidos
# ------------------------------------------------------------------
TARGET = "disease"

VALID_CATEGORIES: dict[str, list[str]] = {
    "sex": ["Male", "Female"],
    "chest_pain": ["typical", "nontypical", "nonanginal", "asymptomatic"],
    "rest_ecg": ["normal", "left ventricular hypertrophy", "ST-T wave abnormality"],
    "thal": ["normal", "fixed", "reversable"],
}

VALID_SLOPE = [1.0, 2.0, 3.0]
VALID_CA = [0.0, 1.0, 2.0, 3.0]
VALID_BOOLEAN = [0.0, 1.0]
EXPECTED_COLUMNS = {
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
    "disease",
}
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "rest_bp": (50, 250),
    "chol": (50, 700),
    "max_hr": (50, 250),
    "old_peak": (0, 10),
}
MAX_NULL_PERCENT: dict[str, float] = {
    "age": 0.0,
    "sex": 0.0,
    "chest_pain": 0.0,
    "rest_ecg": 0.0,
    "thal": 0.0,
    "disease": 0.0,
    "rest_bp": 0.0,
    "chol": 0.0,
    "max_hr": 0.0,
    "old_peak": 0.0,
    "slope": 0.05,
    "ca": 0.1,
    "fbs": 0.05,
    "exang": 0.05,
}
KEY_COLUMNS = [
    "age",
    "sex",
    "chest_pain",
    "rest_bp",
    "chol",
    "max_hr",
    "old_peak",
    "slope",
    "ca",
    "thal",
    "fbs",
    "exang",
    "disease",
]

DEFAULT_INPUT_PATH = Path("data/01_raw/corazon.csv")
DEFAULT_OUTPUT_PATH = Path("data/02_intermediate/corazon_type_fixed.parquet")


class DataValidationError(ValueError):
    """Error de validación de datos previo al guardado de features."""


# ------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------
def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Lee el dataset crudo desde un archivo CSV.

    Args:
        input_path: ruta al archivo CSV con los datos originales.

    Returns:
        DataFrame con los datos sin procesar.
    """
    logger.info("Leyendo datos crudos desde %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Datos cargados: %s filas, %s columnas", df.shape[0], df.shape[1])
    return df


# ------------------------------------------------------------------
# Limpieza de la variable objetivo
# ------------------------------------------------------------------
def clean_target(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Elimina filas con valores nulos o inválidos en la variable objetivo.

    Args:
        df: DataFrame de entrada.
        target: nombre de la columna objetivo.

    Returns:
        DataFrame sin filas inválidas en `target`, con la columna convertida a int.
    """
    df = df.copy()
    n_antes = len(df)

    df = df.dropna(subset=[target])
    df = df[df[target].isin([True, False, 0, 1, "0", "1"])]

    n_eliminadas = n_antes - len(df)
    logger.info(
        "Target '%s': %s filas eliminadas (nulos o valores inválidos)", target, n_eliminadas
    )

    df[target] = df[target].astype(int)
    return df


# ------------------------------------------------------------------
# Limpieza de columnas categóricas nominales
# ------------------------------------------------------------------
def clean_categorical_columns(
    df: pd.DataFrame, valid_categories: dict[str, list[str]] = VALID_CATEGORIES
) -> pd.DataFrame:
    """Elimina filas con valores fuera del catálogo permitido en columnas categóricas.

    Args:
        df: DataFrame de entrada.
        valid_categories: diccionario {columna: [valores válidos]}.

    Returns:
        DataFrame sin filas inválidas, con las columnas convertidas a tipo category.
    """
    df = df.copy()
    for col, validos in valid_categories.items():
        if col not in df.columns:
            logger.warning("Columna '%s' no encontrada en el dataset, se omite", col)
            continue

        mask_invalido = ~df[col].astype(str).isin([str(v) for v in validos]) & df[col].notna()
        n_invalidos = mask_invalido.sum()
        if n_invalidos > 0:
            logger.info("Columna '%s': %s filas inválidas eliminadas", col, n_invalidos)

        df = df[~mask_invalido]
        df[col] = df[col].astype("category")

    return df


# ------------------------------------------------------------------
# Limpieza de columnas ordinales/numéricas discretas
# ------------------------------------------------------------------
def clean_slope(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte `slope` a numérico y elimina valores fuera de {1, 2, 3}."""
    df = df.copy()
    if "slope" not in df.columns:
        return df

    df["slope"] = pd.to_numeric(df["slope"], errors="coerce")
    mask_invalido = ~df["slope"].isin(VALID_SLOPE)
    n_invalidos = mask_invalido.sum()
    if n_invalidos > 0:
        logger.info("Columna 'slope': %s filas inválidas eliminadas", n_invalidos)

    df = df[~mask_invalido]
    return df


def clean_ca(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte `ca` a numérico y elimina valores fuera de {0, 1, 2, 3}."""
    df = df.copy()
    if "ca" not in df.columns:
        return df

    df["ca"] = pd.to_numeric(df["ca"], errors="coerce")
    mask_invalido = ~df["ca"].isin(VALID_CA) & df["ca"].notna()
    n_invalidos = mask_invalido.sum()
    if n_invalidos > 0:
        logger.info("Columna 'ca': %s filas inválidas eliminadas", n_invalidos)

    df = df[~mask_invalido]
    return df


def clean_boolean_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Convierte columnas booleanas (0/1) y elimina valores inválidos.

    Args:
        df: DataFrame de entrada.
        columns: columnas a limpiar. Por defecto ``["fbs", "exang"]``.
    """
    df = df.copy()
    columns = columns or ["fbs", "exang"]

    for col in columns:
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask_invalido = ~df[col].isin(VALID_BOOLEAN) & df[col].notna()
        n_invalidos = mask_invalido.sum()
        if n_invalidos > 0:
            logger.info("Columna '%s': %s filas inválidas eliminadas", col, n_invalidos)

        df = df[~mask_invalido]

    return df


# ------------------------------------------------------------------
# Pipeline completo de features
# ------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las transformaciones de limpieza sobre el dataset crudo.

    Orden de operaciones:
        1. Limpieza del target.
        2. Limpieza de columnas categóricas nominales.
        3. Limpieza de columnas ordinales/numéricas discretas (slope, ca).
        4. Limpieza de columnas booleanas (fbs, exang).
        5. Eliminación de registros duplicados.

    Args:
        df: DataFrame crudo.

    Returns:
        DataFrame de features limpio, listo para el pipeline de entrenamiento.
    """
    n_inicial = len(df)

    df = clean_target(df)
    df = clean_categorical_columns(df)
    df = clean_slope(df)
    df = clean_ca(df)
    df = clean_boolean_columns(df)

    n_antes_duplicados = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_duplicados = n_antes_duplicados - len(df)
    if n_duplicados > 0:
        logger.info("Registros duplicados eliminados: %s", n_duplicados)

    logger.info(
        "Features construidas: %s filas finales de %s originales (%.2f%% retenido)",
        len(df),
        n_inicial,
        len(df) / n_inicial * 100 if n_inicial else 0,
    )
    return df


# ------------------------------------------------------------------
# Almacenamiento
# ------------------------------------------------------------------
def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el DataFrame de features en formato Parquet.

    Args:
        df: DataFrame de features procesado.
        output_path: ruta destino del archivo Parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("Features guardadas en %s (%s filas)", output_path, len(df))


def validate_features(df: pd.DataFrame) -> None:
    """Valida calidad e integridad del dataset antes de persistir.

    Incluye reglas de:
      - estructura/tipos esperados;
      - rangos numéricos y dominios permitidos;
      - porcentaje máximo de nulos;
      - unicidad de registros clave.
    """
    errors: list[str] = []

    missing_columns = sorted(EXPECTED_COLUMNS.difference(df.columns))
    if missing_columns:
        errors.append(f"Columnas faltantes requeridas: {missing_columns}")

    if df.empty:
        errors.append("El dataset de features está vacío después de la limpieza")

    for col, max_null in MAX_NULL_PERCENT.items():
        if col not in df.columns:
            continue
        null_pct = float(df[col].isna().mean())
        if null_pct > max_null:
            errors.append(
                f"Columna '{col}' excede nulos: {null_pct:.2%} > {max_null:.2%}"
            )

    for col, (min_v, max_v) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        if not is_numeric_dtype(df[col]):
            errors.append(f"Columna '{col}' debe ser numérica")
            continue
        out_of_range = (~df[col].between(min_v, max_v)) & df[col].notna()
        if out_of_range.any():
            errors.append(
                f"Columna '{col}' contiene valores fuera de rango [{min_v}, {max_v}]"
            )

    for col, valid_values in VALID_CATEGORIES.items():
        if col not in df.columns:
            continue
        invalid_mask = ~df[col].astype(str).isin(valid_values) & df[col].notna()
        if invalid_mask.any():
            errors.append(f"Columna '{col}' contiene categorías inválidas")

    if "disease" in df.columns and not set(df["disease"].dropna().unique()).issubset({0, 1}):
        errors.append("Columna 'disease' debe contener únicamente valores {0, 1}")

    for col, valid_values in {"slope": VALID_SLOPE, "ca": VALID_CA, "fbs": VALID_BOOLEAN, "exang": VALID_BOOLEAN}.items():
        if col not in df.columns:
            continue
        invalid_mask = ~df[col].isin(valid_values) & df[col].notna()
        if invalid_mask.any():
            errors.append(f"Columna '{col}' contiene valores inválidos")

    if set(KEY_COLUMNS).issubset(df.columns) and df.duplicated(subset=KEY_COLUMNS).any():
        errors.append("Se encontraron registros duplicados en las columnas clave")

    if errors:
        raise DataValidationError("Validación de features fallida: " + " | ".join(errors))


# ------------------------------------------------------------------
# Orquestación / CLI
# ------------------------------------------------------------------
def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Ejecuta el feature pipeline completo: lectura, transformación y guardado.

    Args:
        input_path: ruta del CSV crudo.
        output_path: ruta de salida del Parquet de features.

    Returns:
        DataFrame de features generado (también persistido en disco).
    """
    df_raw = load_raw_data(input_path)
    df_features = build_features(df_raw)
    validate_features(df_features)
    save_features(df_features, output_path)
    return df_features


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Feature pipeline: lee corazon.csv, limpia y transforma los datos, "
        "y guarda las features resultantes en un archivo Parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Ruta al CSV de datos crudos (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Ruta de salida del Parquet de features (default: {DEFAULT_OUTPUT_PATH})",
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
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
