import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.features.feature_engine import FeatureEngine
from src.config.watchlist import WATCHLIST

engine = FeatureEngine()

raw_folder = Path("data/raw")
processed_folder = Path("data/processed")

processed_folder.mkdir(exist_ok=True)

for symbol in WATCHLIST:

    print(f"\nProcessing {symbol}...")

    file = raw_folder / f"{symbol}_5min.csv"

    if not file.exists():
        print(f"Skipping {symbol} (no data)")
        continue

    df = pd.read_csv(file)

    # Make column names lowercase
    df.columns = [c.lower() for c in df.columns]

    # Keep only the columns we need
    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    df = engine.build(df)

    output = processed_folder / f"{symbol}_features.csv"

    df.to_csv(output, index=False)

    print(f"Saved {output}")

print("\n✅ All features built successfully!")