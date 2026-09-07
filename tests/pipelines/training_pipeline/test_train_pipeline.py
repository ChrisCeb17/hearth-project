"""Pruebas unitarias para train_pipeline.py."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from pipelines.training_pipeline.train_pipeline import (
    build_model,
    evaluate_model,
    load_features_data,
    run_pipeline,
    save_metrics,
    save_model,
    split_features_target,
)


@pytest.fixture
def features_df() -> pd.DataFrame:
    """Features de ejemplo para clasificación binaria de disease."""
    return pd.DataFrame(
        {
            "age": [63, 67, 37, 41, 56, 57, 44, 52, 61, 58],
            "sex": ["Male", "Male", "Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male"],
            "chest_pain": [
                "typical",
                "asymptomatic",
                "nonanginal",
                "nontypical",
                "typical",
                "asymptomatic",
                "nonanginal",
                "typical",
                "nontypical",
                "asymptomatic",
            ],
            "rest_bp": [145, 160, 130, 130, 120, 140, 120, 172, 150, 140],
            "chol": [233, 286, 250, 204, 236, 294, 263, 199, 243, 211],
            "fbs": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "rest_ecg": [
                "normal",
                "left ventricular hypertrophy",
                "normal",
                "normal",
                "normal",
                "left ventricular hypertrophy",
                "normal",
                "normal",
                "left ventricular hypertrophy",
                "normal",
            ],
            "max_hr": [150, 108, 187, 172, 178, 170, 173, 162, 137, 165],
            "exang": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            "old_peak": [2.3, 1.5, 3.5, 1.4, 0.8, 1.2, 0.0, 0.5, 1.0, 1.3],
            "slope": [3.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0],
            "ca": [0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0],
            "thal": ["fixed", "normal", "normal", "normal", "fixed", "reversable", "normal", "fixed", "reversable", "normal"],
            "disease": [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
        }
    )


class TestLoadFeaturesData:
    def test_lee_parquet_correctamente(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        input_path = tmp_path / "features.parquet"
        features_df.to_parquet(input_path, index=False)

        result = load_features_data(input_path)

        pd.testing.assert_frame_equal(result, features_df)


class TestSplitFeaturesTarget:
    def test_separa_features_y_target(self, features_df: pd.DataFrame) -> None:
        x, y = split_features_target(features_df)

        assert "disease" not in x.columns
        assert len(x) == len(features_df)
        assert y.name == "disease"
        assert set(y.unique()) == {0, 1}

    def test_target_faltante_lanza_error(self, features_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            split_features_target(features_df.drop(columns=["disease"]))


class TestTrainingAndMetrics:
    def test_entrena_y_genera_metricas(self, features_df: pd.DataFrame) -> None:
        x, y = split_features_target(features_df)
        model = build_model(x, y)

        metrics = evaluate_model(model, x, y)

        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        for value in metrics.values():
            assert 0.0 <= value <= 1.0


class TestSaveArtifacts:
    def test_guarda_modelo(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        x, y = split_features_target(features_df)
        model = build_model(x, y)
        output_path = tmp_path / "models" / "model.joblib"

        save_model(model, output_path)

        assert output_path.exists()
        loaded_model = joblib.load(output_path)
        assert hasattr(loaded_model, "predict")

    def test_guarda_metricas_json(self, tmp_path: Path) -> None:
        output_path = tmp_path / "reporting" / "metrics.json"
        metrics = {
            "accuracy": 0.9,
            "precision": 0.85,
            "recall": 0.8,
            "f1": 0.82,
            "roc_auc": 0.88,
        }

        save_metrics(metrics, output_path)

        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded == metrics


class TestRunPipeline:
    def test_ejecuta_pipeline_completo(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        input_path = tmp_path / "features.parquet"
        model_output_path = tmp_path / "06_models" / "corazon_model.joblib"
        metrics_output_path = tmp_path / "08_reporting" / "training_metrics.json"
        features_df.to_parquet(input_path, index=False)

        metrics = run_pipeline(
            input_path=input_path,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
            test_size=0.3,
            random_state=7,
        )

        assert model_output_path.exists()
        assert metrics_output_path.exists()
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
