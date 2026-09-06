"""Feature pipeline para el proyecto de predicción de enfermedad cardíaca (corazon.csv).

Lee los datos crudos, limpia tipos y valores inválidos, **valida la calidad, consistencia,
formato e integridad de las features resultantes**, y solo si la validación pasa, las
almacena en un archivo Parquet, listas para ser consumidas por el pipeline de entrenamiento.

Si la validación falla, el script:
    - No persiste ningún archivo de salida.
    - Registra en el log el detalle de los casos de fallo.
    - Lanza `DataValidationError` con un mensaje claro (o termina con código de salida != 0
      si se ejecuta desde la línea de comandos).

Uso:
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py \\
        --input data/01_raw/corazon.csv \\
        --output data/02_intermediate/corazon_type_fixed.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuración de columnas y valores válidos (limpieza)
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

DEFAULT_INPUT_PATH = Path("data/01_raw/corazon.csv")
DEFAULT_OUTPUT_PATH = Path("data/02_intermediate/corazon_type_fixed.parquet")

# Porcentaje máximo de nulos tolerado por columna, post-limpieza.
# Ajustar según la tolerancia real de negocio; valores de referencia tomados de la
# proporción de nulos observada en el dataset crudo (ver EDA, notebook 3-analysis).
MAX_NULL_FRACTION = 0.10

# Tolerancia (en lpm) sobre la fórmula teórica de frecuencia cardíaca máxima (220 - edad),
# usada como regla de integridad entre `age` y `max_hr`.
MAX_HR_TOLERANCE = 15


# ------------------------------------------------------------------
# Excepción de validación
# ------------------------------------------------------------------
class DataValidationError(Exception):
    """Se lanza cuando las features no cumplen las reglas de calidad/integridad."""


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
# Pipeline de limpieza (feature engineering básico)
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
        DataFrame de features limpio, listo para validación.
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
# Reglas de validación (calidad, consistencia, formato, integridad)
# ------------------------------------------------------------------
def _max_null_fraction_check(max_fraction: float = MAX_NULL_FRACTION) -> Check:
    """Construye un Check que falla si el % de nulos de la columna supera `max_fraction`."""

    def _validar(serie: pd.Series) -> bool:
        return bool(serie.isna().mean() <= max_fraction)

    return Check(
        _validar,
        ignore_na=False,
        error=f"porcentaje de nulos mayor al {max_fraction:.0%} permitido",
    )


def _max_hr_fisiologicamente_valido(df: pd.DataFrame) -> pd.Series:
    """Regla de integridad entre campos: `max_hr` no debe superar (220 - age + tolerancia).

    220 - edad es la fórmula clínica estándar para la frecuencia cardíaca máxima teórica.
    Se agrega una tolerancia para no rechazar casos límite legítimos.
    """
    limite = 220 - df["age"] + MAX_HR_TOLERANCE
    return df["max_hr"] <= limite


def _sin_filas_duplicadas(df: pd.DataFrame) -> pd.Series:
    """Regla de unicidad: no deben existir registros completamente duplicados."""
    return ~df.duplicated(keep="first")


def get_features_schema() -> DataFrameSchema:
    """Define el esquema de validación de las features (tipos, rangos, categorías, integridad).

    Returns:
        `DataFrameSchema` de Pandera con todas las reglas de calidad del dataset.
    """
    return DataFrameSchema(
        columns={
            "age": Column(
                float,
                checks=[Check.in_range(0, 120), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "sex": Column(
                str,
                checks=[Check.isin(VALID_CATEGORIES["sex"]), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "chest_pain": Column(
                str,
                checks=[Check.isin(VALID_CATEGORIES["chest_pain"]), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "rest_bp": Column(
                float,
                checks=[Check.in_range(60, 250), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "chol": Column(
                float,
                checks=[Check.in_range(100, 600), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "fbs": Column(
                float,
                checks=[Check.isin(VALID_BOOLEAN), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "rest_ecg": Column(
                str,
                checks=[Check.isin(VALID_CATEGORIES["rest_ecg"]), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "max_hr": Column(
                float,
                checks=[Check.in_range(60, 220), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "exang": Column(
                float,
                checks=[Check.isin(VALID_BOOLEAN), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "old_peak": Column(
                float,
                checks=[Check.in_range(0, 10), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "slope": Column(
                float,
                checks=[Check.isin(VALID_SLOPE), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "ca": Column(
                float,
                checks=[Check.isin(VALID_CA), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            "thal": Column(
                str,
                checks=[Check.isin(VALID_CATEGORIES["thal"]), _max_null_fraction_check()],
                nullable=True,
                coerce=True,
            ),
            TARGET: Column(
                int,
                Check.isin([0, 1]),
                nullable=False,  # el target NUNCA debe tener nulos
                coerce=True,
            ),
        },
        checks=[
            # Integridad entre registros: sin filas completamente duplicadas
            Check(_sin_filas_duplicadas, error="existen filas completamente duplicadas"),
            # Integridad entre campos: max_hr coherente con la edad del paciente
            Check(
                _max_hr_fisiologicamente_valido,
                error=f"max_hr supera (220 - age + {MAX_HR_TOLERANCE}), valor fisiológicamente inconsistente",
            ),
        ],
        strict=False,  # permite columnas adicionales no listadas en el esquema
        coerce=True,
    )


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Valida calidad, consistencia, formato e integridad de las features.

    Ejecuta el esquema de Pandera en modo `lazy=True` para acumular TODOS los errores
    de validación (no solo el primero), y los registra en el log antes de fallar.

    Args:
        df: DataFrame de features ya limpio (salida de `build_features`).

    Returns:
        El mismo DataFrame, tipado según el esquema, si la validación es exitosa.

    Raises:
        DataValidationError: si alguna regla de calidad/integridad no se cumple.
            En ese caso, NO se debe persistir el DataFrame.
    """
    schema = get_features_schema()
    try:
        df_validado = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "Validación de datos FALLIDA. Detalle de casos de fallo:\n%s",
            exc.failure_cases.to_string(),
        )
        mensaje = (
            f"Los datos no pasaron la validación de calidad/integridad: "
            f"{len(exc.failure_cases)} caso(s) de fallo encontrados. "
            "Revisa el log para el detalle por columna/regla. No se persistieron features."
        )
        raise DataValidationError(mensaje) from exc

    logger.info(
        "Validación de datos EXITOSA: %s filas, %s columnas cumplen el esquema",
        df_validado.shape[0],
        df_validado.shape[1],
    )
    return df_validado


# ------------------------------------------------------------------
# Almacenamiento
# ------------------------------------------------------------------
def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el DataFrame de features en formato Parquet.

    Solo debe llamarse DESPUÉS de una validación exitosa (`validate_features`).

    Args:
        df: DataFrame de features procesado y validado.
        output_path: ruta destino del archivo Parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("Features guardadas en %s (%s filas)", output_path, len(df))


# ------------------------------------------------------------------
# Orquestación / CLI
# ------------------------------------------------------------------
def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Ejecuta el feature pipeline completo: lectura, transformación, validación y guardado.

    Si la validación falla, la excepción se propaga y `save_features` NUNCA se ejecuta,
    garantizando que no se persistan features de mala calidad.

    Args:
        input_path: ruta del CSV crudo.
        output_path: ruta de salida del Parquet de features.

    Returns:
        DataFrame de features validado (también persistido en disco).

    Raises:
        DataValidationError: si las features no cumplen las reglas de calidad/integridad.
    """
    df_raw = load_raw_data(input_path)
    df_features = build_features(df_raw)
    df_validado = validate_features(df_features)
    save_features(df_validado, output_path)
    return df_validado


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Feature pipeline: lee corazon.csv, limpia, valida y guarda las features "
        "resultantes en un archivo Parquet."
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
    try:
        run_pipeline(args.input, args.output)
    except DataValidationError as exc:
        logger.error("Pipeline detenido: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
