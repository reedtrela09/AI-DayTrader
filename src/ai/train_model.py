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
from sklearn.model_selection import train_test_split

from src.config.features import FEATURE_COLUMNS


class AITrainer:

    def train(self, csv_file):

        print("Loading training data...")

        df = pd.read_csv(csv_file)

        required_columns = FEATURE_COLUMNS + ["Target"]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Training data is missing these columns: "
                + ", ".join(missing_columns)
            )

        df = df.dropna(subset=required_columns)

        X = df[FEATURE_COLUMNS]
        y = df["Target"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            shuffle=False,
        )

        print("Training AI...")

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print("\n========== MODEL REPORT ==========")
        print(f"Accuracy : {accuracy_score(y_test, predictions):.2%}")
        print(
            f"Precision: "
            f"{precision_score(y_test, predictions, zero_division=0):.2%}"
        )
        print(
            f"Recall   : "
            f"{recall_score(y_test, predictions, zero_division=0):.2%}"
        )
        print(
            f"F1 Score : "
            f"{f1_score(y_test, predictions, zero_division=0):.2%}"
        )

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, predictions))

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

        model_path = model_folder / "random_forest.pkl"

        joblib.dump(model, model_path)

        print(f"\nModel saved to:\n{model_path}")