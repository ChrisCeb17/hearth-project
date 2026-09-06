"""Pruebas unitarias para feature_pipeline.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pipelines.feature_pipeline.feature_pipeline import (
    build_features,
    clean_boolean_columns,
    clean_ca,
    clean_categorical_columns,
    clean_slope,
    clean_target,
    load_raw_data,
    save_features,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """DataFrame de ejemplo que imitando el dataset corazon.csv, incluyendo ruido."""
    return pd.DataFrame(
        {
            "age": [63, 45, 52, 41, 60],
            "sex": ["Male", "Female", "Male", "Female", "invalido"],
            "chest_pain": [
                "typical",
                "asymptomatic",
                "nonanginal",
                "nontypical",
                "typical",
            ],
            "rest_bp": [145, 130, 120, 110, 140],
            "chol": [233, 204, 199, 250, 300],
            "fbs": [1, 0, "0", "no_valido", 1],
            "rest_ecg": ["normal", "ST-T wave abnormality", "normal", "normal", "raro"],
            "max_hr": [150, 172, 168, 180, 140],
            "exang": [0, 0, 1, "1", 0],
            "old_peak": [2.3, 1.4, 0.0, 0.5, 1.2],
            "slope": ["1", "2", "3", "no_numero", "2"],
            "ca": ["0", "1", "5", "2", "0"],
            "thal": ["fixed", "normal", "reversable", "normal", "desconocido"],
            "disease": [1, 0, 0, None, "no_valido"],
        }
    )


class TestCleanTarget:
    def test_elimina_nulos(self, raw_df: pd.DataFrame) -> None:
        result = clean_target(raw_df)
        assert result["disease"].isna().sum() == 0

    def test_elimina_valores_invalidos(self, raw_df: pd.DataFrame) -> None:
        result = clean_target(raw_df)
        assert set(result["disease"].unique()).issubset({0, 1})

    def test_convierte_a_entero(self, raw_df: pd.DataFrame) -> None:
        result = clean_target(raw_df)
        assert result["disease"].dtype == int

    def test_conserva_filas_validas(self) -> None:
        df = pd.DataFrame({"disease": [0, 1, 0, 1]})
        result = clean_target(df)
        assert len(result) == 4


class TestCleanCategoricalColumns:
    def test_elimina_categoria_invalida_sex(self, raw_df: pd.DataFrame) -> None:
        result = clean_categorical_columns(raw_df)
        assert "invalido" not in result["sex"].astype(str).values

    def test_elimina_categoria_invalida_thal(self, raw_df: pd.DataFrame) -> None:
        result = clean_categorical_columns(raw_df)
        assert "desconocido" not in result["thal"].astype(str).values

    def test_convierte_a_category_dtype(self, raw_df: pd.DataFrame) -> None:
        result = clean_categorical_columns(raw_df)
        assert result["sex"].dtype.name == "category"
        assert result["chest_pain"].dtype.name == "category"

    def test_columna_faltante_no_rompe(self) -> None:
        df = pd.DataFrame({"otra_columna": [1, 2, 3]})
        result = clean_categorical_columns(df)
        assert len(result) == 3


class TestCleanSlope:
    def test_elimina_valores_no_numericos(self, raw_df: pd.DataFrame) -> None:
        result = clean_slope(raw_df)
        assert "no_numero" not in result["slope"].astype(str).values

    def test_valores_validos_se_conservan(self) -> None:
        df = pd.DataFrame({"slope": ["1", "2", "3"]})
        result = clean_slope(df)
        assert len(result) == 3
        assert set(result["slope"].unique()) == {1.0, 2.0, 3.0}


class TestCleanCa:
    def test_elimina_valores_fuera_de_rango(self, raw_df: pd.DataFrame) -> None:
        result = clean_ca(raw_df)
        assert 5.0 not in result["ca"].values

    def test_valores_validos_se_conservan(self) -> None:
        df = pd.DataFrame({"ca": ["0", "1", "2", "3"]})
        result = clean_ca(df)
        assert len(result) == 4


class TestCleanBooleanColumns:
    def test_elimina_valores_invalidos(self, raw_df: pd.DataFrame) -> None:
        result = clean_boolean_columns(raw_df)
        assert "no_valido" not in result["fbs"].astype(str).values

    def test_valores_validos_se_conservan(self) -> None:
        df = pd.DataFrame({"fbs": [0, 1, "0", "1"]})
        result = clean_boolean_columns(df, columns=["fbs"])
        assert len(result) == 4
        assert set(result["fbs"].unique()) == {0.0, 1.0}


class TestBuildFeatures:
    def test_pipeline_completo_elimina_filas_invalidas(
        self, raw_df: pd.DataFrame
    ) -> None:
        result = build_features(raw_df)
        assert len(result) < len(raw_df)
        assert result["disease"].isna().sum() == 0

    def test_pipeline_elimina_duplicados(self) -> None:
        df = pd.DataFrame(
            {
                "sex": ["Male", "Male"],
                "chest_pain": ["typical", "typical"],
                "rest_ecg": ["normal", "normal"],
                "thal": ["normal", "normal"],
                "slope": ["1", "1"],
                "ca": ["0", "0"],
                "fbs": [0, 0],
                "exang": [0, 0],
                "disease": [1, 1],
            }
        )
        result = build_features(df)
        assert len(result) == 1

    def test_resultado_no_tiene_nulos_en_target(self, raw_df: pd.DataFrame) -> None:
        result = build_features(raw_df)
        assert result["disease"].isna().sum() == 0


class TestLoadRawData:
    def test_lee_csv_correctamente(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test_data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

        result = load_raw_data(csv_path)

        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_archivo_inexistente_lanza_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_raw_data(tmp_path / "no_existe.csv")


class TestSaveFeatures:
    def test_guarda_parquet_correctamente(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        output_path = tmp_path / "subdir" / "features.parquet"

        save_features(df, output_path)

        assert output_path.exists()
        result = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(result, df)

    def test_crea_directorios_faltantes(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1]})
        output_path = tmp_path / "no_existe" / "aun" / "features.parquet"

        save_features(df, output_path)

        assert output_path.parent.exists()
