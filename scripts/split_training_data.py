from pathlib import Path

import pandas as pd

from src.config.watchlist import WATCHLIST


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"
SPLIT_FOLDER = PROCESSED_FOLDER / "splits"

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20


def split_symbol(symbol: str) -> tuple[int, int, int]:
    input_file = PROCESSED_FOLDER / f"{symbol}_training.csv"

    if not input_file.exists():
        print(f"Skipping {symbol}: training file not found.")
        return 0, 0, 0

    df = pd.read_csv(input_file)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{symbol}_training.csv is missing the timestamp column."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(subset=["timestamp", "Target"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    total_rows = len(df)

    if total_rows < 100:
        print(f"Skipping {symbol}: only {total_rows} usable rows.")
        return 0, 0, 0

    train_end = int(total_rows * TRAIN_RATIO)
    validation_end = int(
        total_rows * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    symbol_folder = SPLIT_FOLDER / symbol
    symbol_folder.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(symbol_folder / "train.csv", index=False)
    validation_df.to_csv(
        symbol_folder / "validation.csv",
        index=False,
    )
    test_df.to_csv(symbol_folder / "test.csv", index=False)

    print(
        f"{symbol:<6} "
        f"Train: {len(train_df):>5} | "
        f"Validation: {len(validation_df):>5} | "
        f"Test: {len(test_df):>5}"
    )

    return len(train_df), len(validation_df), len(test_df)


def combine_split(split_name: str) -> None:
    frames = []

    for symbol in WATCHLIST:
        split_file = SPLIT_FOLDER / symbol / f"{split_name}.csv"

        if not split_file.exists():
            continue

        df = pd.read_csv(split_file)
        df["Symbol"] = symbol
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No files were available for the {split_name} split."
        )

    combined = pd.concat(frames, ignore_index=True)

    output_file = SPLIT_FOLDER / f"combined_{split_name}.csv"
    combined.to_csv(output_file, index=False)

    print(
        f"Saved {output_file.name}: "
        f"{len(combined):,} rows"
    )


def main() -> None:
    SPLIT_FOLDER.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("TIME-BASED TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 72)

    for symbol in WATCHLIST:
        split_symbol(symbol)

    print()
    combine_split("train")
    combine_split("validation")
    combine_split("test")

    print()
    print(f"Split files saved to:\n{SPLIT_FOLDER}")


if __name__ == "__main__":
    main()