import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.ai.label_data import LabelGenerator
from src.config.watchlist import WATCHLIST

generator = LabelGenerator()

processed = Path("data/processed")

for symbol in WATCHLIST:

    feature_file = processed / f"{symbol}_features.csv"

    if not feature_file.exists():
        print(f"Skipping {symbol}")
        continue

    print(f"Labeling {symbol}...")

    df = pd.read_csv(feature_file)

    df = generator.generate(df)

    output = processed / f"{symbol}_training.csv"

    df.to_csv(output, index=False)

    print(f"Saved {output}")

print("\n✅ Finished labeling all stocks!")