from pathlib import Path

import joblib
import pandas as pd

from src.config.features import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_training.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest.pkl"
)

RESULTS_FOLDER = PROJECT_ROOT / "results"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]

# These match the current label generator:
# take profit = 1.5 ATR
# stop loss = 1.0 ATR
WIN_R = 1.50
LOSS_R = -1.00

# Estimated slippage/fees expressed in R per trade.
TRADING_COST_R = 0.02


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from cumulative R returns."""

    equity_curve = returns.cumsum()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve - running_peak

    return float(drawdown.min())


def run_backtest() -> None:
    print("Loading model and training data...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + ["Target"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            + ", ".join(missing_columns)
        )

    df = df.dropna(subset=required_columns).reset_index(drop=True)

    # Use the final 20% as out-of-sample test data.
    split_index = int(len(df) * 0.80)

    test_df = df.iloc[split_index:].copy()

    if test_df.empty:
        raise ValueError("The test dataset is empty.")

    probabilities = model.predict_proba(
        test_df[FEATURE_COLUMNS]
    )[:, 1]

    test_df["Probability"] = probabilities

    summary_rows = []

    print()
    print("=" * 75)
    print("AI PROBABILITY BACKTEST")
    print("=" * 75)
    print(
        f"Out-of-sample rows: {len(test_df):,}"
    )
    print()

    for threshold in THRESHOLDS:
        trades = test_df[
            test_df["Probability"] >= threshold
        ].copy()

        if trades.empty:
            summary_rows.append(
                {
                    "Threshold": threshold,
                    "Trades": 0,
                    "Wins": 0,
                    "Losses": 0,
                    "WinRate": 0.0,
                    "NetR": 0.0,
                    "ProfitFactor": 0.0,
                    "ExpectancyR": 0.0,
                    "MaxDrawdownR": 0.0,
                }
            )
            continue

        trades["TradeR"] = trades["Target"].apply(
            lambda target: WIN_R if int(target) == 1 else LOSS_R
        )

        trades["TradeR"] = (
            trades["TradeR"] - TRADING_COST_R
        )

        wins = trades[trades["Target"] == 1]
        losses = trades[trades["Target"] == 0]

        gross_profit = trades.loc[
            trades["TradeR"] > 0,
            "TradeR",
        ].sum()

        gross_loss = abs(
            trades.loc[
                trades["TradeR"] < 0,
                "TradeR",
            ].sum()
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
        )

        net_r = trades["TradeR"].sum()
        expectancy_r = trades["TradeR"].mean()
        max_drawdown_r = calculate_max_drawdown(
            trades["TradeR"]
        )

        summary_rows.append(
            {
                "Threshold": threshold,
                "Trades": len(trades),
                "Wins": len(wins),
                "Losses": len(losses),
                "WinRate": len(wins) / len(trades),
                "NetR": net_r,
                "ProfitFactor": profit_factor,
                "ExpectancyR": expectancy_r,
                "MaxDrawdownR": max_drawdown_r,
            }
        )

        trade_log_path = (
            RESULTS_FOLDER
            / f"trades_threshold_{int(threshold * 100)}.csv"
        )

        trades.to_csv(trade_log_path, index=False)

    summary = pd.DataFrame(summary_rows)

    print(
        summary.to_string(
            index=False,
            formatters={
                "Threshold": lambda value: f"{value:.0%}",
                "WinRate": lambda value: f"{value:.2%}",
                "NetR": lambda value: f"{value:.2f}",
                "ProfitFactor": lambda value: f"{value:.2f}",
                "ExpectancyR": lambda value: f"{value:.3f}",
                "MaxDrawdownR": lambda value: f"{value:.2f}",
            },
        )
    )

    summary_path = (
        RESULTS_FOLDER
        / "probability_backtest_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    valid_results = summary[summary["Trades"] > 0]

    if not valid_results.empty:
        best_result = valid_results.loc[
            valid_results["NetR"].idxmax()
        ]

        print()
        print("=" * 75)
        print("BEST THRESHOLD BY NET R")
        print("=" * 75)
        print(
            f"Threshold    : "
            f"{best_result['Threshold']:.0%}"
        )
        print(
            f"Trades       : "
            f"{int(best_result['Trades'])}"
        )
        print(
            f"Win Rate     : "
            f"{best_result['WinRate']:.2%}"
        )
        print(
            f"Net R        : "
            f"{best_result['NetR']:.2f}"
        )
        print(
            f"Profit Factor: "
            f"{best_result['ProfitFactor']:.2f}"
        )
        print(
            f"Expectancy   : "
            f"{best_result['ExpectancyR']:.3f}R"
        )
        print(
            f"Max Drawdown : "
            f"{best_result['MaxDrawdownR']:.2f}R"
        )

    print()
    print(f"Results saved to: {RESULTS_FOLDER}")


if __name__ == "__main__":
    run_backtest()