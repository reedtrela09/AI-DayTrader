from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config.features import FEATURE_COLUMNS


class AITrainer:

    def train(
        self,
        train_csv: str,
        validation_csv: str,
    ) -> None:

        print("Loading training and validation data...")

        train_df = pd.read_csv(train_csv)
        validation_df = pd.read_csv(validation_csv)

        required_columns = FEATURE_COLUMNS + ["Target"]

        for name, df in [
            ("training", train_df),
            ("validation", validation_df),
        ]:
            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"{name.title()} data is missing columns: "
                    + ", ".join(missing_columns)
                )

        train_df = train_df.dropna(
            subset=required_columns
        ).reset_index(drop=True)

        validation_df = validation_df.dropna(
            subset=required_columns
        ).reset_index(drop=True)

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["Target"].astype(int)

        X_validation = validation_df[FEATURE_COLUMNS]
        y_validation = validation_df["Target"].astype(int)

        print(f"Training rows: {len(train_df):,}")
        print(f"Validation rows: {len(validation_df):,}")
        print("Training AI...")

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        print("\n========== VALIDATION REPORT ==========")
        print(
            f"Accuracy : "
            f"{accuracy_score(y_validation, predictions):.2%}"
        )
        print(
            f"Precision: "
            f"{precision_score(y_validation, predictions, zero_division=0):.2%}"
        )
        print(
            f"Recall   : "
            f"{recall_score(y_validation, predictions, zero_division=0):.2%}"
        )
        print(
            f"F1 Score : "
            f"{f1_score(y_validation, predictions, zero_division=0):.2%}"
        )

        print("\nConfusion Matrix")
        print(confusion_matrix(y_validation, predictions))

        print("\nFeature Importance")

        importance = sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )

        for feature, score in importance:
            print(f"{feature:<15} {score:.3f}")

        project_root = Path(__file__).resolve().parents[2]
        model_folder = project_root / "models"
        model_folder.mkdir(parents=True, exist_ok=True)

        model_path = (
            model_folder
            / "random_forest_validation.pkl"
        )

        joblib.dump(model, model_path)

        print(f"\nModel saved to:\n{model_path}")