"""Feature pipeline para transformar datos crudos en features de modelado."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS = [
    "age",
    "rest_bp",
    "chol",
    "fbs",
    "max_hr",
    "exang",
    "old_peak",
    "slope",
    "ca",
]
CATEGORICAL_COLUMNS = ["sex", "chest_pain", "rest_ecg", "thal"]
TARGET_COLUMN = "disease"


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia dataset crudo y lo convierte en features numéricos + label."""
    work_df = df.copy()

    for column in CATEGORICAL_COLUMNS:
        mode = work_df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        work_df[column] = (
            work_df[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA})
            .fillna(fill_value)
        )

    for column in [*NUMERIC_COLUMNS, TARGET_COLUMN]:
        work_df[column] = pd.to_numeric(work_df[column], errors="coerce")

    work_df = work_df[work_df[TARGET_COLUMN].isin([0, 1])].copy()

    for column in NUMERIC_COLUMNS:
        work_df[column] = work_df[column].fillna(work_df[column].median())

    encoded_categoricals = pd.get_dummies(work_df[CATEGORICAL_COLUMNS], dtype=int)
    result_df = pd.concat(
        [work_df[NUMERIC_COLUMNS].reset_index(drop=True), encoded_categoricals.reset_index(drop=True)],
        axis=1,
    )
    result_df[TARGET_COLUMN] = work_df[TARGET_COLUMN].astype(int).reset_index(drop=True)
    return result_df


def build_feature_pipeline(input_path: Path | str, output_path: Path | str) -> Path:
    """Lee archivo crudo, crea features y guarda resultado en CSV."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    raw_df = pd.read_csv(in_path)
    feature_df = transform_features(raw_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)
    return out_path


def _default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    input_path = repo_root / "data" / "01_raw" / "corazon.csv"
    output_path = repo_root / "data" / "04_feature" / "corazon_features.csv"
    return input_path, output_path


def main() -> None:
    """Entry point ejecutable para correr el pipeline de features."""
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Pipeline de creación de features")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Ruta del CSV de entrada",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Ruta del CSV de salida",
    )
    args = parser.parse_args()
    build_feature_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
