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

from src.config.settings import (
    BUY_THRESHOLD,
    DRY_RUN,
    WATCH_THRESHOLD,
)
from src.config.watchlist import WATCHLIST
from src.scanner.scanner import AIScanner


PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"
LOGS_FOLDER = PROJECT_ROOT / "logs"

SCAN_JOURNAL = LOGS_FOLDER / "scan_journal.csv"
TRADE_JOURNAL = LOGS_FOLDER / "trade_journal.csv"


st.set_page_config(
    page_title="AI-DayTrader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
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
            "Alpaca API keys were not found in .env."
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
                    "No complete feature rows available."
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
        "Confidence",
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

    return {
        "equity": equity,
        "buying_power": float(account.buying_power),
        "cash": float(account.cash),
        "daily_change": equity - last_equity,
        "market_open": bool(clock.is_open),
        "next_open": clock.next_open,
        "next_close": clock.next_close,
        "positions": positions,
    }


def load_scan_journal() -> pd.DataFrame:
    if not SCAN_JOURNAL.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(SCAN_JOURNAL)

    if "Timestamp" in dataframe.columns:
        dataframe["Timestamp"] = pd.to_datetime(
            dataframe["Timestamp"],
            errors="coerce",
        )

    return dataframe


def load_trade_journal() -> pd.DataFrame:
    if not TRADE_JOURNAL.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(TRADE_JOURNAL)

    if "Timestamp" in dataframe.columns:
        dataframe["Timestamp"] = pd.to_datetime(
            dataframe["Timestamp"],
            errors="coerce",
        )

    return dataframe


def build_positions_dataframe(
    positions,
) -> pd.DataFrame:
    rows = []

    for position in positions:
        rows.append(
            {
                "Symbol": position.symbol,
                "Quantity": float(position.qty),
                "Average Entry": float(
                    position.avg_entry_price
                ),
                "Current Price": float(
                    position.current_price
                ),
                "Market Value": float(
                    position.market_value
                ),
                "Unrealized P/L": float(
                    position.unrealized_pl
                ),
                "Unrealized P/L %": (
                    float(position.unrealized_plpc) * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def format_rankings(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
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


def render_sidebar(
    account: dict,
    scan_journal: pd.DataFrame,
    trade_journal: pd.DataFrame,
) -> None:
    with st.sidebar:
        st.header("System Status")

        if account["market_open"]:
            st.success("Market: OPEN")
        else:
            st.warning("Market: CLOSED")

        st.success("Alpaca: Connected")
        st.success("AI model: Loaded")

        if DRY_RUN:
            st.info("Mode: DRY RUN")
        else:
            st.error("Mode: ORDER EXECUTION")

        st.divider()

        st.subheader("Configuration")

        st.write(
            f"Buy threshold: **{BUY_THRESHOLD:.0%}**"
        )
        st.write(
            f"Watch threshold: **{WATCH_THRESHOLD:.0%}**"
        )
        st.write(
            f"Watchlist symbols: **{len(WATCHLIST)}**"
        )

        st.divider()

        st.subheader("Journal Status")

        st.write(
            f"Logged scans: **{len(scan_journal):,}**"
        )
        st.write(
            f"Trade decisions: **{len(trade_journal):,}**"
        )


def render_overview(
    account: dict,
    rankings: pd.DataFrame,
) -> None:
    st.subheader("Account Overview")

    columns = st.columns(5)

    columns[0].metric(
        "Market",
        "OPEN" if account["market_open"] else "CLOSED",
    )

    columns[1].metric(
        "Account Equity",
        f"${account['equity']:,.2f}",
        f"${account['daily_change']:,.2f} today",
    )

    columns[2].metric(
        "Buying Power",
        f"${account['buying_power']:,.2f}",
    )

    columns[3].metric(
        "Cash",
        f"${account['cash']:,.2f}",
    )

    columns[4].metric(
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
    st.subheader("Top AI Opportunity")

    valid_rankings = rankings.dropna(
        subset=["Confidence"]
    )

    if valid_rankings.empty:
        st.warning("No valid AI rankings are available.")
        return

    top_pick = valid_rankings.iloc[0]
    top_columns = st.columns(4)

    top_columns[0].metric(
        "Symbol",
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

    top_columns[3].metric(
        "Last Price",
        f"${top_pick['Last Price']:,.2f}",
    )


def render_scanner(
    rankings: pd.DataFrame,
    scan_journal: pd.DataFrame,
) -> None:
    st.subheader("Current AI Rankings")

    st.dataframe(
        format_rankings(rankings),
        use_container_width=True,
        hide_index=False,
    )

    st.subheader("Current Confidence")

    chart_dataframe = rankings.dropna(
        subset=["Confidence"]
    )[["Symbol", "Confidence"]]

    if chart_dataframe.empty:
        st.write("No confidence data available.")
    else:
        st.bar_chart(
            chart_dataframe.set_index("Symbol")
        )

    st.divider()
    st.subheader("Confidence History")

    if scan_journal.empty:
        st.write("No scan history has been logged yet.")
        return

    history = scan_journal.copy()

    history = history.dropna(
        subset=["Timestamp", "Symbol", "Confidence"]
    )

    selected_symbols = st.multiselect(
        "Symbols to chart",
        options=WATCHLIST,
        default=WATCHLIST[:3],
    )

    if selected_symbols:
        history = history[
            history["Symbol"].isin(selected_symbols)
        ]

    if history.empty:
        st.write(
            "No confidence history matches the selection."
        )
        return

    pivot = history.pivot_table(
        index="Timestamp",
        columns="Symbol",
        values="Confidence",
        aggfunc="last",
    ).sort_index()

    st.line_chart(pivot)


def render_scan_journal(
    scan_journal: pd.DataFrame,
) -> None:
    st.subheader("Recent Scan Journal")

    if scan_journal.empty:
        st.write("No scan records exist yet.")
        return

    rows_to_show = st.slider(
        "Number of scan records",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )

    recent = scan_journal.sort_values(
        "Timestamp",
        ascending=False,
    ).head(rows_to_show)

    display = recent.copy()

    if "Confidence" in display.columns:
        display["Confidence"] = (
            display["Confidence"] * 100
        ).round(2)

        display = display.rename(
            columns={
                "Confidence": "Confidence %",
            }
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    action_counts = (
        scan_journal["Action"]
        .value_counts()
        .rename_axis("Action")
        .to_frame("Count")
    )

    st.subheader("Signal Totals")
    st.bar_chart(action_counts)


def render_trade_journal(
    trade_journal: pd.DataFrame,
) -> None:
    st.subheader("Dry-Run Trade Journal")

    if trade_journal.empty:
        st.info(
            "No BUY-level trade decisions have been "
            "recorded yet. This page will populate when "
            "a symbol reaches the buy threshold."
        )
        return

    recent = trade_journal.sort_values(
        "Timestamp",
        ascending=False,
    )

    display = recent.copy()

    if "Confidence" in display.columns:
        display["Confidence"] = (
            display["Confidence"] * 100
        ).round(2)

        display = display.rename(
            columns={
                "Confidence": "Confidence %",
            }
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    approved_count = int(
        trade_journal["Approved"].astype(bool).sum()
    )

    rejected_count = (
        len(trade_journal) - approved_count
    )

    metrics = st.columns(3)

    metrics[0].metric(
        "Total Decisions",
        len(trade_journal),
    )

    metrics[1].metric(
        "Approved",
        approved_count,
    )

    metrics[2].metric(
        "Rejected",
        rejected_count,
    )


def render_positions(account: dict) -> None:
    st.subheader("Open Paper Positions")

    positions_dataframe = build_positions_dataframe(
        account["positions"]
    )

    if positions_dataframe.empty:
        st.info("No open paper positions.")
        return

    st.dataframe(
        positions_dataframe,
        use_container_width=True,
        hide_index=True,
    )


def render_dashboard() -> None:
    scanner = load_scanner()
    trading_client = load_trading_client()

    account = load_account_information(
        trading_client
    )

    rankings = load_rankings(scanner)
    scan_journal = load_scan_journal()
    trade_journal = load_trade_journal()

    render_sidebar(
        account=account,
        scan_journal=scan_journal,
        trade_journal=trade_journal,
    )

    st.title("📈 AI-DayTrader Dashboard")

    st.caption(
        "Alpaca paper account • "
        f"{'Dry-run mode' if DRY_RUN else 'Execution mode'} • "
        f"Updated {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    tabs = st.tabs(
        [
            "Overview",
            "AI Scanner",
            "Scan Journal",
            "Trade Journal",
            "Positions",
        ]
    )

    with tabs[0]:
        render_overview(
            account=account,
            rankings=rankings,
        )

    with tabs[1]:
        render_scanner(
            rankings=rankings,
            scan_journal=scan_journal,
        )

    with tabs[2]:
        render_scan_journal(
            scan_journal=scan_journal,
        )

    with tabs[3]:
        render_trade_journal(
            trade_journal=trade_journal,
        )

    with tabs[4]:
        render_positions(account)

    st.divider()

    if DRY_RUN:
        st.warning(
            "Dry-run mode is enabled. "
            "No orders are being submitted."
        )
    else:
        st.error(
            "Order execution mode is enabled."
        )


@st.fragment(run_every="30s")
def live_dashboard() -> None:
    render_dashboard()


live_dashboard()