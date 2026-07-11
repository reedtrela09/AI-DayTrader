from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


class AITrainer:

    def train(self, csv_file):

        print("Loading training data...")

        df = pd.read_csv(csv_file).dropna()

        feature_columns = [
    "EMA9",
    "EMA21",
    "EMA50",
    "EMA200",
    "EMA9_EMA21",
    "EMA21_EMA50",
    "DistEMA200",
    "RSI",
    "MACD",
    "MACDSignal",
    "MACDHist",
    "ROC",
    "ATR",
    "ATRPercent",
    "BBUpper",
    "BBLower",
    "BBWidth",
    "VolumeSMA20",
    "RelativeVolume",
    "VWAP",
    "VWAPDistance",
    "Body",
    "UpperWick",
    "LowerWick",
    "BodyPercent",
    "Hour",
    "Minute",
    "Return",
]
        

        X = df[feature_columns]
        y = df["Target"]

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
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print("\n========== MODEL REPORT ==========")
        print(f"Accuracy : {accuracy_score(y_test, predictions):.2%}")
        print(f"Precision: {precision_score(y_test, predictions):.2%}")
        print(f"Recall   : {recall_score(y_test, predictions):.2%}")
        print(f"F1 Score : {f1_score(y_test, predictions):.2%}")

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, predictions))

        print("\nFeature Importance")
        importance = sorted(
            zip(feature_columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )

        for feature, score in importance:
            print(f"{feature:<12} {score:.3f}")

        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "models" / "random_forest.pkl"

        joblib.dump(model, model_path)

        print(f"\nModel saved to:\n{model_path}")