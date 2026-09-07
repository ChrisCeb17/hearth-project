"""Pruebas unitarias para train_pipeline.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from pipelines.training_pipeline.train_pipeline import (
    TrainTestValidationError,
    analyze_generalization,
    build_model,
    build_preprocessor,
    compare_train_cv_test,
    cross_validate_model,
    evaluate_model,
    get_cv_splitter,
    load_features,
    plot_cv_scores,
    run_pipeline,
    save_metrics,
    save_model,
    split_train_test,
    train_model,
    validate_train_test_split,
)


@pytest.fixture
def features_df() -> pd.DataFrame:
    """DataFrame sintético de features ya limpias, con estructura similar a corazon.csv."""
    rng = np.random.default_rng(42)
    n = 60

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


# ------------------------------------------------------------------
# Lectura de datos
# ------------------------------------------------------------------
class TestLoadFeatures:
    def test_lee_parquet_correctamente(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        parquet_path = tmp_path / "features.parquet"
        features_df.to_parquet(parquet_path)

        result = load_features(parquet_path)

        assert len(result) == len(features_df)
        assert "disease" in result.columns

    def test_archivo_inexistente_lanza_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_features(tmp_path / "no_existe.parquet")

    def test_target_faltante_lanza_error(self, tmp_path: Path) -> None:
        df_sin_target = pd.DataFrame({"age": [1, 2, 3]})
        parquet_path = tmp_path / "sin_target.parquet"
        df_sin_target.to_parquet(parquet_path)

        with pytest.raises(ValueError, match="disease"):
            load_features(parquet_path)


# ------------------------------------------------------------------
# Train / Test split
# ------------------------------------------------------------------
class TestSplitTrainTest:
    def test_proporciones_correctas(self, features_df: pd.DataFrame) -> None:
        x_train, x_test, _y_train, _y_test = split_train_test(
            features_df, test_size=0.2, random_state=42
        )
        total = len(features_df)
        assert len(x_test) == pytest.approx(total * 0.2, abs=2)
        assert len(x_train) + len(x_test) == total

    def test_target_no_incluido_en_x(self, features_df: pd.DataFrame) -> None:
        x_train, x_test, _, _ = split_train_test(features_df)
        assert "disease" not in x_train.columns
        assert "disease" not in x_test.columns

    def test_estratificacion_mantiene_proporcion_de_clases(self, features_df: pd.DataFrame) -> None:
        _, _, y_train, _y_test = split_train_test(features_df, test_size=0.3, random_state=42)
        prop_original = features_df["disease"].mean()
        prop_train = y_train.mean()
        assert abs(prop_train - prop_original) < 0.15  # noqa: PLR2004

    def test_reproducible_con_misma_semilla(self, features_df: pd.DataFrame) -> None:
        split_a = split_train_test(features_df, random_state=1)
        split_b = split_train_test(features_df, random_state=1)
        pd.testing.assert_frame_equal(split_a[0], split_b[0])


# ------------------------------------------------------------------
# Construcción de pipeline y entrenamiento
# ------------------------------------------------------------------
class TestBuildPreprocessorAndModel:
    def test_build_preprocessor_incluye_columnas_esperadas(self) -> None:
        preprocessor = build_preprocessor(numeric_columns=["age"], categorical_columns=["sex"])
        nombres_transformers = [t[0] for t in preprocessor.transformers]
        assert "numeric" in nombres_transformers
        assert "categoric" in nombres_transformers

    def test_build_model_aplica_hiperparametros(self) -> None:
        model = build_model({"n_estimators": 10, "max_depth": 3, "random_state": 0})
        assert model.n_estimators == 10  # noqa: PLR2004
        assert model.max_depth == 3  # noqa: PLR2004


class TestTrainModel:
    def test_devuelve_pipeline_ajustado(self, features_df: pd.DataFrame) -> None:
        x_train, _, y_train, _ = split_train_test(features_df)
        pipeline = train_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}
        )

        assert isinstance(pipeline, Pipeline)
        # Un pipeline ajustado debe poder predecir sin error
        preds = pipeline.predict(x_train)
        assert len(preds) == len(x_train)

    def test_pipeline_predice_solo_clases_validas(self, features_df: pd.DataFrame) -> None:
        x_train, _, y_train, _ = split_train_test(features_df)
        pipeline = train_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}
        )
        preds = pipeline.predict(x_train)
        assert set(np.unique(preds)).issubset({0, 1})


# ------------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------------
class TestEvaluateModel:
    def test_metricas_contienen_claves_esperadas(self, features_df: pd.DataFrame) -> None:
        x_train, x_test, y_train, y_test = split_train_test(features_df)
        pipeline = train_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}
        )
        metrics = evaluate_model(pipeline, x_test, y_test)

        claves_esperadas = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "confusion_matrix",
            "n_test_samples",
            "roc_auc",
        }
        assert claves_esperadas.issubset(metrics.keys())

    def test_metricas_en_rango_valido(self, features_df: pd.DataFrame) -> None:
        x_train, x_test, y_train, y_test = split_train_test(features_df)
        pipeline = train_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}
        )
        metrics = evaluate_model(pipeline, x_test, y_test)

        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0.0 <= metrics[metric_name] <= 1.0, f"{metric_name} fuera de [0, 1]"

    def test_n_test_samples_coincide(self, features_df: pd.DataFrame) -> None:
        x_train, x_test, y_train, y_test = split_train_test(features_df)
        pipeline = train_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}
        )
        metrics = evaluate_model(pipeline, x_test, y_test)
        assert metrics["n_test_samples"] == len(y_test)

    def test_confusion_matrix_es_2x2_para_problema_binario(self, features_df: pd.DataFrame) -> None:
        x_train, x_test, y_train, y_test = split_train_test(features_df)
        pipeline = train_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}
        )
        metrics = evaluate_model(pipeline, x_test, y_test)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2  # noqa: PLR2004
        assert all(len(fila) == 2 for fila in cm)  # noqa: PLR2004


# ------------------------------------------------------------------
# Almacenamiento
# ------------------------------------------------------------------
class TestSaveModel:
    def test_guarda_y_recarga_pipeline(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        x_train, x_test, y_train, _ = split_train_test(features_df)
        pipeline = train_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}
        )
        output_path = tmp_path / "modelo.joblib"

        save_model(pipeline, output_path)

        assert output_path.exists()
        pipeline_cargado = load(output_path)
        preds_original = pipeline.predict(x_test)
        preds_cargado = pipeline_cargado.predict(x_test)
        np.testing.assert_array_equal(preds_original, preds_cargado)

    def test_crea_directorios_faltantes(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        x_train, _, y_train, _ = split_train_test(features_df)
        pipeline = train_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}
        )
        output_path = tmp_path / "no_existe" / "aun" / "modelo.joblib"

        save_model(pipeline, output_path)

        assert output_path.exists()


class TestSaveMetrics:
    def test_guarda_json_correctamente(self, tmp_path: Path) -> None:
        metrics = {"accuracy": 0.9, "recall": 0.85, "confusion_matrix": [[10, 2], [1, 15]]}
        output_path = tmp_path / "metrics.json"

        save_metrics(metrics, output_path)

        assert output_path.exists()

        with output_path.open() as f:
            metrics_leidas = json.load(f)
        assert metrics_leidas == metrics

    def test_crea_directorios_faltantes(self, tmp_path: Path) -> None:
        output_path = tmp_path / "no_existe" / "metrics.json"
        save_metrics({"accuracy": 0.9}, output_path)
        assert output_path.exists()


# ------------------------------------------------------------------
# Integración: pipeline completo de punta a punta
# ------------------------------------------------------------------
class TestRunPipeline:
    def test_genera_modelo_y_metricas(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        input_path = tmp_path / "features.parquet"
        model_output_path = tmp_path / "modelo.joblib"
        metrics_output_path = tmp_path / "metrics.json"

        features_df.to_parquet(input_path)

        metrics = run_pipeline(
            input_path=input_path,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
            model_params={"n_estimators": 10, "random_state": 42},
        )

        assert model_output_path.exists()
        assert metrics_output_path.exists()
        assert "recall" in metrics

    def test_modelo_guardado_es_utilizable(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        input_path = tmp_path / "features.parquet"
        model_output_path = tmp_path / "modelo.joblib"
        metrics_output_path = tmp_path / "metrics.json"

        features_df.to_parquet(input_path)

        run_pipeline(
            input_path=input_path,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
            model_params={"n_estimators": 10, "random_state": 42},
        )

        modelo = load(model_output_path)
        x_sample = features_df.drop(columns=["disease"]).head(3)
        preds = modelo.predict(x_sample)
        assert len(preds) == 3  # noqa: PLR2004


# ------------------------------------------------------------------
# Validación de la separación train/test (validate_train_test_split)
# ------------------------------------------------------------------
@pytest.fixture
def split_valido(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split train/test estándar, sin problemas, para usar como base en los tests de validación."""
    return split_train_test(features_df, test_size=0.2, random_state=42)


class TestValidateTrainTestSplitCasosValidos:
    def test_split_valido_no_lanza_error(
        self, split_valido: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        x_train, x_test, y_train, y_test = split_valido
        resultado = validate_train_test_split(x_train, x_test, y_train, y_test)
        assert resultado["passed"] is True
        assert resultado["leakage_detected"] is False

    def test_split_valido_sin_errores(
        self, split_valido: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        x_train, x_test, y_train, y_test = split_valido
        resultado = validate_train_test_split(x_train, x_test, y_train, y_test)
        assert resultado["errors"] == []


class TestValidateTrainTestSplitLeakage:
    def test_indices_compartidos_lanza_error(
        self, split_valido: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        x_train, x_test, y_train, y_test = split_valido
        x_test_leak = x_test.copy()
        x_test_leak.index = list(x_train.index[: len(x_test_leak)])

        with pytest.raises(TrainTestValidationError):
            validate_train_test_split(x_train, x_test_leak, y_train, y_test)

    def test_filas_duplicadas_entre_conjuntos_lanza_error(
        self, split_valido: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        x_train, x_test, y_train, y_test = split_valido
        x_test_dup = pd.concat(
            [x_train.iloc[[0]].reset_index(drop=True), x_test.iloc[1:].reset_index(drop=True)],
            ignore_index=True,
        )
        y_test_dup = pd.concat(
            [y_train.iloc[[0]].reset_index(drop=True), y_test.iloc[1:].reset_index(drop=True)],
            ignore_index=True,
        )

        with pytest.raises(TrainTestValidationError):
            validate_train_test_split(x_train, x_test_dup, y_train, y_test_dup)

    def test_mensaje_de_error_es_claro(
        self, split_valido: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        x_train, x_test, y_train, y_test = split_valido
        x_test_leak = x_test.copy()
        x_test_leak.index = list(x_train.index[: len(x_test_leak)])

        with pytest.raises(TrainTestValidationError, match="fuga de información"):
            validate_train_test_split(x_train, x_test_leak, y_train, y_test)


class TestValidateTrainTestSplitAdvertencias:
    def test_categoria_nueva_en_test_genera_advertencia_no_error(
        self, split_valido: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        x_train, x_test, y_train, y_test = split_valido
        x_test_nueva_cat = x_test.copy()
        x_test_nueva_cat.loc[x_test_nueva_cat.index[0], "thal"] = "categoria_nunca_vista"

        resultado = validate_train_test_split(x_train, x_test_nueva_cat, y_train, y_test)

        assert resultado["passed"] is True
        assert any("thal" in w for w in resultado["warnings"])

    def test_drift_numerico_fuerte_genera_advertencia_no_error(
        self, split_valido: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        x_train, x_test, y_train, y_test = split_valido
        x_test_drift = x_test.copy()
        x_test_drift["chol"] = x_test_drift["chol"] + 1000

        resultado = validate_train_test_split(x_train, x_test_drift, y_train, y_test)

        assert resultado["passed"] is True
        assert any("chol" in w for w in resultado["warnings"])

    def test_test_muy_pequeno_genera_advertencia(self, features_df: pd.DataFrame) -> None:
        x_train, x_test, y_train, y_test = split_train_test(
            features_df, test_size=0.02, random_state=42
        )
        resultado = validate_train_test_split(x_train, x_test, y_train, y_test)
        assert any("pequeño" in w for w in resultado["warnings"])


class TestRunPipelineDetieneSiHayLeakage:
    def test_run_pipeline_no_entrena_si_hay_leakage(
        self, tmp_path: Path, features_df: pd.DataFrame
    ) -> None:
        # Duplicar el dataframe completo fuerza filas idénticas entre train y test
        df_con_duplicados = pd.concat([features_df, features_df], ignore_index=True)

        input_path = tmp_path / "features.parquet"
        model_output_path = tmp_path / "modelo.joblib"
        metrics_output_path = tmp_path / "metrics.json"
        df_con_duplicados.to_parquet(input_path)

        with pytest.raises(TrainTestValidationError):
            run_pipeline(
                input_path=input_path,
                model_output_path=model_output_path,
                metrics_output_path=metrics_output_path,
                model_params={"n_estimators": 10, "random_state": 42},
            )

        assert not model_output_path.exists()
        assert not metrics_output_path.exists()


# ------------------------------------------------------------------
# Validación robusta del modelo: cross-validation, comparación, diagnóstico
# ------------------------------------------------------------------
class TestGetCvSplitter:
    def test_devuelve_stratified_kfold(self) -> None:
        splitter = get_cv_splitter(n_splits=5)
        assert isinstance(splitter, StratifiedKFold)
        assert splitter.n_splits == 5

    def test_shuffle_activado_para_reproducibilidad(self) -> None:
        splitter = get_cv_splitter()
        assert splitter.shuffle is True
        assert splitter.random_state is not None


class TestCrossValidateModel:
    def test_devuelve_metricas_esperadas(self, features_df: pd.DataFrame) -> None:
        x_train, _, y_train, _ = split_train_test(features_df)
        resultado = cross_validate_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}, cv_folds=3
        )

        for metrica in ["accuracy", "precision", "recall", "f1"]:
            assert metrica in resultado
            assert "mean" in resultado[metrica]
            assert "std" in resultado[metrica]
            assert "scores" in resultado[metrica]

    def test_numero_de_scores_coincide_con_folds(self, features_df: pd.DataFrame) -> None:
        x_train, _, y_train, _ = split_train_test(features_df)
        resultado = cross_validate_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}, cv_folds=4
        )

        assert len(resultado["accuracy"]["scores"]) == 4

    def test_scores_en_rango_valido(self, features_df: pd.DataFrame) -> None:
        x_train, _, y_train, _ = split_train_test(features_df)
        resultado = cross_validate_model(
            x_train, y_train, model_params={"n_estimators": 10, "random_state": 42}, cv_folds=3
        )

        for metrica, valores in resultado.items():
            assert 0.0 <= valores["mean"] <= 1.0, f"{metrica} fuera de rango"
            for score in valores["scores"]:
                assert 0.0 <= score <= 1.0


class TestCompareTrainCvTest:
    def test_estructura_de_comparacion(self) -> None:
        train_metrics = {"accuracy": 0.9, "recall": 0.85}
        cv_metrics = {
            "accuracy": {"mean": 0.8, "std": 0.05, "scores": [0.75, 0.8, 0.85]},
            "recall": {"mean": 0.75, "std": 0.03, "scores": [0.72, 0.75, 0.78]},
        }
        test_metrics = {"accuracy": 0.82, "recall": 0.77}

        resultado = compare_train_cv_test(
            train_metrics, cv_metrics, test_metrics, metrics_to_compare=["accuracy", "recall"]
        )

        assert resultado["accuracy"]["train"] == 0.9
        assert resultado["accuracy"]["cv_mean"] == 0.8
        assert resultado["accuracy"]["cv_std"] == 0.05
        assert resultado["accuracy"]["test"] == 0.82


class TestAnalyzeGeneralization:
    def test_detecta_overfitting(self) -> None:
        comparison = {
            "recall": {"train": 1.0, "cv_mean": 0.5, "cv_std": 0.1, "test": 0.55},
        }
        resultado = analyze_generalization(comparison, primary_metric="recall")

        assert resultado["status"] == "overfitting"
        assert len(resultado["recommendations"]) > 0

    def test_detecta_underfitting(self) -> None:
        comparison = {
            "recall": {"train": 0.4, "cv_mean": 0.38, "cv_std": 0.05, "test": 0.42},
        }
        resultado = analyze_generalization(comparison, primary_metric="recall")

        assert resultado["status"] == "underfitting"
        assert len(resultado["recommendations"]) > 0

    def test_detecta_buena_generalizacion(self) -> None:
        comparison = {
            "recall": {"train": 0.82, "cv_mean": 0.80, "cv_std": 0.03, "test": 0.79},
        }
        resultado = analyze_generalization(comparison, primary_metric="recall")

        assert resultado["status"] == "buena_generalizacion"

    def test_calcula_brechas_correctamente(self) -> None:
        comparison = {
            "recall": {"train": 0.90, "cv_mean": 0.70, "cv_std": 0.05, "test": 0.65},
        }
        resultado = analyze_generalization(comparison, primary_metric="recall")

        assert resultado["gap_train_cv"] == pytest.approx(0.20)
        assert resultado["gap_train_test"] == pytest.approx(0.25)


class TestPlotCvScores:
    def test_genera_archivo_png(self, tmp_path: Path) -> None:
        cv_metrics = {
            "accuracy": {"mean": 0.8, "std": 0.05, "scores": [0.75, 0.8, 0.85, 0.78, 0.82]},
            "recall": {"mean": 0.75, "std": 0.03, "scores": [0.72, 0.75, 0.78, 0.74, 0.76]},
        }
        output_path = tmp_path / "cv_plot.png"

        plot_cv_scores(cv_metrics, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_crea_directorios_faltantes(self, tmp_path: Path) -> None:
        cv_metrics = {"accuracy": {"mean": 0.8, "std": 0.05, "scores": [0.75, 0.8, 0.85]}}
        output_path = tmp_path / "no_existe" / "cv_plot.png"

        plot_cv_scores(cv_metrics, output_path)

        assert output_path.exists()


class TestRunPipelineIncluyeValidacionDeModelo:
    def test_reporte_incluye_todas_las_secciones(
        self, tmp_path: Path, features_df: pd.DataFrame
    ) -> None:
        input_path = tmp_path / "features.parquet"
        model_output_path = tmp_path / "modelo.joblib"
        metrics_output_path = tmp_path / "metrics.json"
        features_df.to_parquet(input_path)

        reporte = run_pipeline(
            input_path=input_path,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
            model_params={"n_estimators": 10, "random_state": 42},
        )

        assert "train_metrics" in reporte
        assert "cross_validation" in reporte
        assert "train_cv_test_comparison" in reporte
        assert "generalization_analysis" in reporte
        # Compatibilidad hacia atrás: las métricas de test siguen en el nivel superior
        assert "recall" in reporte

    def test_genera_grafica_de_cv(self, tmp_path: Path, features_df: pd.DataFrame) -> None:
        input_path = tmp_path / "features.parquet"
        model_output_path = tmp_path / "modelo.joblib"
        metrics_output_path = tmp_path / "metrics.json"
        features_df.to_parquet(input_path)

        run_pipeline(
            input_path=input_path,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
            model_params={"n_estimators": 10, "random_state": 42},
        )

        cv_plot_path = metrics_output_path.with_name(metrics_output_path.stem + "_cv_boxplot.png")
        assert cv_plot_path.exists()

    def test_metrics_json_es_deserializable(
        self, tmp_path: Path, features_df: pd.DataFrame
    ) -> None:
        input_path = tmp_path / "features.parquet"
        model_output_path = tmp_path / "modelo.joblib"
        metrics_output_path = tmp_path / "metrics.json"
        features_df.to_parquet(input_path)

        run_pipeline(
            input_path=input_path,
            model_output_path=model_output_path,
            metrics_output_path=metrics_output_path,
            model_params={"n_estimators": 10, "random_state": 42},
        )

        with metrics_output_path.open() as f:
            contenido = json.load(f)
        assert "generalization_analysis" in contenido
