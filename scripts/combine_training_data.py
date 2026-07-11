import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.config.watchlist import WATCHLIST

processed = Path("data/processed")

frames = []

for symbol in WATCHLIST:

    file = processed / f"{symbol}_training.csv"

    if not file.exists():
        print(f"Skipping {symbol}")
        continue

    df = pd.read_csv(file)

    df["Symbol"] = symbol

    frames.append(df)

combined = pd.concat(frames, ignore_index=True)

combined.to_csv(
    processed / "combined_training.csv",
    index=False
)

print()
print(f"Combined {len(frames)} symbols.")
print(f"Rows: {len(combined):,}")
print("Saved combined_training.csv")