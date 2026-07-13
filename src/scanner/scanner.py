from pathlib import Path

import joblib
import pandas as pd

from src.config.features import FEATURE_COLUMNS


class AIScanner:

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        model_path = (
            project_root
            / "models"
            / "random_forest_validation.pkl"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file was not found: {model_path}"
            )

        self.model = joblib.load(model_path)

    def score(self, df: pd.DataFrame) -> float:
        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Scanner data is missing these columns: "
                + ", ".join(missing_columns)
            )

        clean_df = df.dropna(subset=FEATURE_COLUMNS)

        if clean_df.empty:
            raise ValueError(
                "No complete rows are available for scoring."
            )

        latest = clean_df.iloc[-1:]

        probability = self.model.predict_proba(
            latest[FEATURE_COLUMNS]
        )[0][1]

        return float(probability)