import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config.watchlist import WATCHLIST
from src.scanner.scanner import AIScanner


scanner = AIScanner()

processed_folder = project_root / "data" / "processed"
results = []

print()
print("=========================================")
print(" AI DAYTRADER SCANNER")
print("=========================================")
print()
print(f"Scanning {len(WATCHLIST)} stocks...")
print()

for symbol in WATCHLIST:
    file_path = processed_folder / f"{symbol}_features.csv"

    if not file_path.exists():
        print(f"Skipping {symbol}: feature file not found.")
        continue

    try:
        df = pd.read_csv(file_path)
        confidence = scanner.score(df)

        results.append(
            {
                "symbol": symbol,
                "confidence": confidence,
            }
        )

    except Exception as error:
        print(f"Skipping {symbol}: {error}")

results.sort(
    key=lambda item: item["confidence"],
    reverse=True,
)

print()
print("Rank  Symbol  Confidence  Action")
print("--------------------------------")

for rank, result in enumerate(results, start=1):
    confidence = result["confidence"]

    if confidence >= 0.70:
        action = "BUY"
    elif confidence >= 0.55:
        action = "WATCH"
    else:
        action = "SKIP"

    print(
        f"{rank:<5} "
        f"{result['symbol']:<7} "
        f"{confidence:>9.2%}  "
        f"{action}"
    )

print()
print("=========================================")