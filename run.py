from pathlib import Path

import pandas as pd

from src.config.watchlist import WATCHLIST
from src.data.alpaca_data import AlpacaDataManager
from src.features.feature_engine import FeatureEngine
from src.scanner.scanner import AIScanner


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"


def get_action(confidence: float) -> str:
    if confidence >= 0.75:
        return "BUY"

    if confidence >= 0.60:
        return "WATCH"

    return "HOLD"


def refresh_symbol(
    symbol: str,
    data_manager: AlpacaDataManager,
    feature_engine: FeatureEngine,
) -> pd.DataFrame:
    print(f"Refreshing {symbol}...")

    df = data_manager.download_5min(
        symbol=symbol,
        days=30,
    )

    if df.empty:
        raise ValueError(f"No market data was returned for {symbol}.")

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]

    required_raw_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_raw_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{symbol} data is missing columns: "
            + ", ".join(missing_columns)
        )

    RAW_FOLDER.mkdir(parents=True, exist_ok=True)
    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

    raw_file = RAW_FOLDER / f"{symbol}_5min.csv"
    df.to_csv(raw_file, index=False)

    features_df = feature_engine.build(df)

    processed_file = (
        PROCESSED_FOLDER
        / f"{symbol}_features.csv"
    )

    features_df.to_csv(
        processed_file,
        index=False,
    )

    return features_df


def display_results(results: list[dict]) -> None:
    results.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    print()
    print("=" * 60)
    print(" AI DAYTRADER RESULTS")
    print("=" * 60)
    print()
    print(
        f"{'Rank':<6}"
        f"{'Symbol':<9}"
        f"{'Confidence':<14}"
        f"{'Action'}"
    )
    print("-" * 45)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank:<6}"
            f"{result['symbol']:<9}"
            f"{result['confidence']:<14.2%}"
            f"{result['action']}"
        )

    print()
    print("=" * 60)


def main() -> None:
    print()
    print("=" * 60)
    print(" AI DAYTRADER")
    print("=" * 60)
    print()
    print("Refreshing market data and rebuilding features...")
    print()

    data_manager = AlpacaDataManager()
    feature_engine = FeatureEngine()
    scanner = AIScanner()

    results = []

    for symbol in WATCHLIST:
        try:
            features_df = refresh_symbol(
                symbol=symbol,
                data_manager=data_manager,
                feature_engine=feature_engine,
            )

            confidence = scanner.score(features_df)

            results.append(
                {
                    "symbol": symbol,
                    "confidence": confidence,
                    "action": get_action(confidence),
                }
            )

            print(
                f"Finished {symbol}: "
                f"{confidence:.2%}"
            )

        except Exception as error:
            print(
                f"Could not process {symbol}: "
                f"{error}"
            )

    if not results:
        print()
        print("No symbols were successfully scanned.")
        return

    display_results(results)


if __name__ == "__main__":
    main()