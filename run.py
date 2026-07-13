from pathlib import Path
import pandas as pd

from src.config.watchlist import WATCHLIST
from src.scanner.scanner import AIScanner


def main():

    project_root = Path(__file__).resolve().parent
    processed_folder = project_root / "data" / "processed"

    scanner = AIScanner()

    results = []

    print("=" * 60)
    print(" AI DAYTRADER")
    print("=" * 60)
    print()

    print(f"Scanning {len(WATCHLIST)} symbols...\n")

    for symbol in WATCHLIST:

        csv_file = processed_folder / f"{symbol}_features.csv"

        if not csv_file.exists():
            print(f"{symbol:<6} Missing feature file")
            continue

        try:
            df = pd.read_csv(csv_file)

            confidence = scanner.score(df)

            if confidence >= 0.75:
                action = "BUY"
            elif confidence >= 0.60:
                action = "WATCH"
            else:
                action = "HOLD"

            results.append(
                (
                    symbol,
                    confidence,
                    action,
                )
            )

        except Exception as error:
            print(f"{symbol}: {error}")

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    print()

    print(f"{'Rank':<5}{'Symbol':<8}{'Confidence':<14}Action")
    print("-" * 45)

    for rank, (symbol, confidence, action) in enumerate(results, start=1):

        print(
            f"{rank:<5}"
            f"{symbol:<8}"
            f"{confidence:>8.2%}      "
            f"{action}"
        )


if __name__ == "__main__":
    main()