from pathlib import Path

import joblib
import pandas as pd

from src.config.features import FEATURE_COLUMNS
from src.config.watchlist import WATCHLIST


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"
RESULTS_FOLDER = PROJECT_ROOT / "results" / "realistic_backtest"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]

TEST_PERCENT = 0.20
TAKE_PROFIT_ATR = 1.50
STOP_LOSS_ATR = 1.00
MAX_HOLDING_BARS = 12

# Estimated slippage on entry and exit.
SLIPPAGE_PERCENT = 0.0001


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model was not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_symbol_data(symbol: str) -> pd.DataFrame:
    file_path = PROCESSED_FOLDER / f"{symbol}_training.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training file was not found for {symbol}:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = FEATURE_COLUMNS + [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "ATR",
        "Target",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{symbol} is missing columns: "
            + ", ".join(missing_columns)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=required_columns)
    df = df.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(df) * (1 - TEST_PERCENT))

    return df.iloc[split_index:].reset_index(drop=True)


def simulate_symbol(
    df: pd.DataFrame,
    symbol: str,
    model,
    threshold: float,
) -> list[dict]:
    if len(df) < MAX_HOLDING_BARS + 2:
        return []

    probabilities = model.predict_proba(
        df[FEATURE_COLUMNS]
    )[:, 1]

    df = df.copy()
    df["Probability"] = probabilities

    trades = []
    index = 0

    while index < len(df) - MAX_HOLDING_BARS - 1:
        signal_row = df.iloc[index]
        confidence = float(signal_row["Probability"])

        if confidence < threshold:
            index += 1
            continue

        # Enter on the next candle instead of the signal candle.
        entry_index = index + 1
        entry_row = df.iloc[entry_index]

        raw_entry_price = float(entry_row["open"])
        atr = float(signal_row["ATR"])

        if atr <= 0:
            index += 1
            continue

        # Apply unfavorable entry slippage.
        entry_price = raw_entry_price * (1 + SLIPPAGE_PERCENT)

        stop_price = entry_price - (atr * STOP_LOSS_ATR)
        target_price = entry_price + (atr * TAKE_PROFIT_ATR)

        exit_price = None
        exit_reason = None
        exit_index = None

        final_index = min(
            entry_index + MAX_HOLDING_BARS,
            len(df) - 1,
        )

        for future_index in range(
            entry_index,
            final_index + 1,
        ):
            future_row = df.iloc[future_index]

            candle_high = float(future_row["high"])
            candle_low = float(future_row["low"])

            stop_hit = candle_low <= stop_price
            target_hit = candle_high >= target_price

            # Conservative assumption:
            # if both are touched in the same candle,
            # count the stop as occurring first.
            if stop_hit:
                exit_price = stop_price
                exit_reason = "STOP"
                exit_index = future_index
                break

            if target_hit:
                exit_price = target_price
                exit_reason = "TARGET"
                exit_index = future_index
                break

        if exit_price is None:
            exit_index = final_index
            exit_price = float(df.iloc[exit_index]["close"])
            exit_reason = "TIME"

        # Apply unfavorable exit slippage.
        exit_price *= 1 - SLIPPAGE_PERCENT

        risk_per_share = entry_price - stop_price
        pnl_per_share = exit_price - entry_price

        trade_r = (
            pnl_per_share / risk_per_share
            if risk_per_share > 0
            else 0.0
        )

        trades.append(
            {
                "Symbol": symbol,
                "SignalTime": signal_row["timestamp"],
                "EntryTime": entry_row["timestamp"],
                "ExitTime": df.iloc[exit_index]["timestamp"],
                "Confidence": confidence,
                "EntryPrice": entry_price,
                "StopPrice": stop_price,
                "TargetPrice": target_price,
                "ExitPrice": exit_price,
                "ExitReason": exit_reason,
                "BarsHeld": exit_index - entry_index + 1,
                "TradeR": trade_r,
            }
        )

        # Skip forward so trades cannot overlap for this symbol.
        index = exit_index + 1

    return trades


def calculate_max_drawdown(trade_returns: pd.Series) -> float:
    equity_curve = trade_returns.cumsum()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve - running_peak

    if drawdown.empty:
        return 0.0

    return float(drawdown.min())


def summarize_trades(
    trades: pd.DataFrame,
    threshold: float,
) -> dict:
    if trades.empty:
        return {
            "Threshold": threshold,
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "WinRate": 0.0,
            "NetR": 0.0,
            "ProfitFactor": 0.0,
            "ExpectancyR": 0.0,
            "MaxDrawdownR": 0.0,
            "Targets": 0,
            "Stops": 0,
            "TimeExits": 0,
        }

    trades = trades.sort_values("ExitTime").reset_index(drop=True)

    wins = trades[trades["TradeR"] > 0]
    losses = trades[trades["TradeR"] <= 0]

    gross_profit = wins["TradeR"].sum()
    gross_loss = abs(losses["TradeR"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    return {
        "Threshold": threshold,
        "Trades": len(trades),
        "Wins": len(wins),
        "Losses": len(losses),
        "WinRate": len(wins) / len(trades),
        "NetR": trades["TradeR"].sum(),
        "ProfitFactor": profit_factor,
        "ExpectancyR": trades["TradeR"].mean(),
        "MaxDrawdownR": calculate_max_drawdown(
            trades["TradeR"]
        ),
        "Targets": int(
            (trades["ExitReason"] == "TARGET").sum()
        ),
        "Stops": int(
            (trades["ExitReason"] == "STOP").sum()
        ),
        "TimeExits": int(
            (trades["ExitReason"] == "TIME").sum()
        ),
    }


def run_backtest():
    print("Loading AI model...")

    model = load_model()
    summary_rows = []

    print()
    print("=" * 88)
    print("REALISTIC AI BACKTEST")
    print("=" * 88)

    for threshold in THRESHOLDS:
        print(f"\nTesting threshold {threshold:.0%}...")

        all_trades = []

        for symbol in WATCHLIST:
            try:
                symbol_df = load_symbol_data(symbol)

                symbol_trades = simulate_symbol(
                    df=symbol_df,
                    symbol=symbol,
                    model=model,
                    threshold=threshold,
                )

                all_trades.extend(symbol_trades)

                print(
                    f"  {symbol:<6} "
                    f"{len(symbol_trades):>4} trades"
                )

            except Exception as error:
                print(f"  Skipping {symbol}: {error}")

        trades_df = pd.DataFrame(all_trades)

        if not trades_df.empty:
            trades_df = trades_df.sort_values(
                "ExitTime"
            ).reset_index(drop=True)

        summary = summarize_trades(
            trades=trades_df,
            threshold=threshold,
        )

        summary_rows.append(summary)

        trade_file = (
            RESULTS_FOLDER
            / f"trades_{int(threshold * 100)}.csv"
        )

        trades_df.to_csv(trade_file, index=False)

    summary_df = pd.DataFrame(summary_rows)

    print()
    print("=" * 88)
    print("BACKTEST SUMMARY")
    print("=" * 88)

    print(
        summary_df.to_string(
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

    summary_file = RESULTS_FOLDER / "summary.csv"
    summary_df.to_csv(summary_file, index=False)

    valid_results = summary_df[
        summary_df["Trades"] >= 30
    ]

    if not valid_results.empty:
        best = valid_results.loc[
            valid_results["ProfitFactor"].idxmax()
        ]

        print()
        print("=" * 88)
        print("BEST THRESHOLD WITH AT LEAST 30 TRADES")
        print("=" * 88)
        print(f"Threshold    : {best['Threshold']:.0%}")
        print(f"Trades       : {int(best['Trades'])}")
        print(f"Win Rate     : {best['WinRate']:.2%}")
        print(f"Net R        : {best['NetR']:.2f}")
        print(
            f"Profit Factor: "
            f"{best['ProfitFactor']:.2f}"
        )
        print(
            f"Expectancy   : "
            f"{best['ExpectancyR']:.3f}R"
        )
        print(
            f"Max Drawdown : "
            f"{best['MaxDrawdownR']:.2f}R"
        )

    print()
    print(f"Results saved to:\n{RESULTS_FOLDER}")


if __name__ == "__main__":
    run_backtest()