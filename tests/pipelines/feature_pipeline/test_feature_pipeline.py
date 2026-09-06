from pathlib import Path

import pandas as pd

from pipelines.feature_pipeline.feature_pipeline import build_feature_pipeline, transform_features


def test_transform_features_filters_invalid_target_and_encodes_categoricals() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "age": "63",
                "sex": "Male",
                "chest_pain": "typical",
                "rest_bp": "145",
                "chol": "233",
                "fbs": "1",
                "rest_ecg": "left ventricular hypertrophy ",
                "max_hr": "150",
                "exang": "0",
                "old_peak": "2.3",
                "slope": "3",
                "ca": "0",
                "thal": "fixed",
                "disease": "0",
            },
            {
                "age": "invalid",
                "sex": "Female",
                "chest_pain": "asymptomatic",
                "rest_bp": "120",
                "chol": "229",
                "fbs": "0",
                "rest_ecg": "normal",
                "max_hr": "129",
                "exang": "1",
                "old_peak": "2.6",
                "slope": "2",
                "ca": "2",
                "thal": "reversable",
                "disease": "gsfdg",
            },
        ]
    )

    features = transform_features(raw_df)

    assert features.shape[0] == 1
    assert features["disease"].tolist() == [0]
    assert "sex_Male" in features.columns
    assert "rest_ecg_left ventricular hypertrophy" in features.columns


def test_build_feature_pipeline_writes_output_file(tmp_path: Path) -> None:
    input_path = tmp_path / "corazon_raw.csv"
    output_path = tmp_path / "out" / "corazon_features.csv"
    pd.DataFrame(
        [
            {
                "age": 37,
                "sex": "Male",
                "chest_pain": "nonanginal",
                "rest_bp": 130,
                "chol": 250,
                "fbs": 0,
                "rest_ecg": "normal",
                "max_hr": 187,
                "exang": 0,
                "old_peak": 3.5,
                "slope": 3,
                "ca": 0,
                "thal": "normal",
                "disease": 0,
            }
        ]
    ).to_csv(input_path, index=False)

    saved_path = build_feature_pipeline(input_path, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    written_df = pd.read_csv(output_path)
    assert written_df["disease"].tolist() == [0]
