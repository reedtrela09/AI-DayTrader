import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.features.feature_engine import FeatureEngine

print("Loading data...")

df = pd.read_csv("data/raw/SPY_5min.csv")

engine = FeatureEngine()

df = engine.build(df)

print(df.tail())

df.to_csv("data/processed/SPY_features.csv", index=False)

print()
print("✅ Features saved!")