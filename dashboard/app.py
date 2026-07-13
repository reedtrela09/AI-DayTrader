import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import BUY_THRESHOLD, WATCH_THRESHOLD
from src.config.watchlist import WATCHLIST
from src.scanner.scanner import AIScanner


PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"


st.set_page_config(
    page_title="AI-DayTrader",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def load_scanner() -> AIScanner:
    return AIScanner()


@st.cache_resource
def load_trading_client() -> TradingClient:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API keys were not found in the .env file."
        )

    return TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
    )


def get_action(confidence: float) -> str:
    if confidence >= BUY_THRESHOLD:
        return "BUY"

    if confidence >= WATCH_THRESHOLD:
        return "WATCH"

    return "HOLD"


def load_rankings(scanner: AIScanner) -> pd.DataFrame:
    rows = []

    for symbol in WATCHLIST:
        feature_file = (
            PROCESSED_FOLDER
            / f"{symbol}_features.csv"
        )

        if not feature_file.exists():
            rows.append(
                {
                    "Symbol": symbol,
                    "Confidence": None,
                    "Action": "NO DATA",
                    "Last Price": None,
                    "ATR": None,
                }
            )
            continue

        try:
            dataframe = pd.read_csv(feature_file)

            confidence = scanner.score(dataframe)

            clean_dataframe = dataframe.dropna()

            if clean_dataframe.empty:
                raise ValueError(
                    "No complete feature rows were available."
                )

            latest = clean_dataframe.iloc[-1]

            rows.append(
                {
                    "Symbol": symbol,
                    "Confidence": confidence,
                    "Action": get_action(confidence),
                    "Last Price": float(latest["close"]),
                    "ATR": float(latest["ATR"]),
                }
            )

        except Exception:
            rows.append(
                {
                    "Symbol": symbol,
                    "Confidence": None,
                    "Action": "ERROR",
                    "Last Price": None,
                    "ATR": None,
                }
            )

    rankings = pd.DataFrame(rows)

    rankings = rankings.sort_values(
        by="Confidence",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    rankings.index = rankings.index + 1
    rankings.index.name = "Rank"

    return rankings


def load_account_information(
    trading_client: TradingClient,
) -> dict:
    account = trading_client.get_account()
    clock = trading_client.get_clock()
    positions = trading_client.get_all_positions()

    equity = float(account.equity)
    last_equity = float(account.last_equity)
    daily_change = equity - last_equity

    return {
        "equity": equity,
        "buying_power": float(account.buying_power),
        "cash": float(account.cash),
        "daily_change": daily_change,
        "market_open": bool(clock.is_open),
        "next_open": clock.next_open,
        "next_close": clock.next_close,
        "positions": positions,
    }


def build_positions_dataframe(positions) -> pd.DataFrame:
    rows = []

    for position in positions:
        rows.append(
            {
                "Symbol": position.symbol,
                "Quantity": float(position.qty),
                "Average Entry": float(position.avg_entry_price),
                "Current Price": float(position.current_price),
                "Market Value": float(position.market_value),
                "Unrealized P/L": float(position.unrealized_pl),
                "Unrealized P/L %": (
                    float(position.unrealized_plpc) * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def format_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    formatted = rankings.copy()

    formatted["Confidence"] = formatted[
        "Confidence"
    ].apply(
        lambda value: (
            f"{value:.2%}"
            if pd.notna(value)
            else ""
        )
    )

    formatted["Last Price"] = formatted[
        "Last Price"
    ].apply(
        lambda value: (
            f"${value:,.2f}"
            if pd.notna(value)
            else ""
        )
    )

    formatted["ATR"] = formatted["ATR"].apply(
        lambda value: (
            f"${value:,.2f}"
            if pd.notna(value)
            else ""
        )
    )

    return formatted


def render_dashboard() -> None:
    scanner = load_scanner()
    trading_client = load_trading_client()

    account = load_account_information(
        trading_client
    )

    rankings = load_rankings(scanner)

    st.title("📈 AI-DayTrader Dashboard")

    st.caption(
        "Paper account • Dry-run mode • "
        f"Updated {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    market_status = (
        "OPEN"
        if account["market_open"]
        else "CLOSED"
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Market",
        market_status,
    )

    metric_columns[1].metric(
        "Account Equity",
        f"${account['equity']:,.2f}",
        f"${account['daily_change']:,.2f} today",
    )

    metric_columns[2].metric(
        "Buying Power",
        f"${account['buying_power']:,.2f}",
    )

    metric_columns[3].metric(
        "Cash",
        f"${account['cash']:,.2f}",
    )

    metric_columns[4].metric(
        "Open Positions",
        len(account["positions"]),
    )

    if account["market_open"]:
        st.info(
            f"Next market close: "
            f"{account['next_close']}"
        )
    else:
        st.info(
            f"Next market open: "
            f"{account['next_open']}"
        )

    st.divider()

    st.subheader("AI Rankings")

    top_valid = rankings.dropna(
        subset=["Confidence"]
    )

    if not top_valid.empty:
        top_pick = top_valid.iloc[0]

        top_columns = st.columns(3)

        top_columns[0].metric(
            "Top Symbol",
            top_pick["Symbol"],
        )

        top_columns[1].metric(
            "Confidence",
            f"{top_pick['Confidence']:.2%}",
        )

        top_columns[2].metric(
            "Action",
            top_pick["Action"],
        )

    st.dataframe(
        format_rankings(rankings),
        use_container_width=True,
        hide_index=False,
    )

    st.divider()

    st.subheader("Confidence Chart")

    chart_dataframe = rankings.dropna(
        subset=["Confidence"]
    )[["Symbol", "Confidence"]]

    if not chart_dataframe.empty:
        chart_dataframe = chart_dataframe.set_index(
            "Symbol"
        )

        st.bar_chart(chart_dataframe)

    st.divider()

    st.subheader("Open Paper Positions")

    positions_dataframe = build_positions_dataframe(
        account["positions"]
    )

    if positions_dataframe.empty:
        st.write("No open paper positions.")
    else:
        st.dataframe(
            positions_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "Dry-run mode is enabled. "
        "This dashboard does not submit orders."
    )


@st.fragment(run_every="30s")
def live_dashboard() -> None:
    render_dashboard()


live_dashboard()