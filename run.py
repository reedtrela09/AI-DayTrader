import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from src.config.settings import (
    ATR_STOP_MULTIPLIER,
    ATR_TARGET_MULTIPLIER,
    BUY_THRESHOLD,
    DRY_RUN,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PERCENT,
    RISK_PER_TRADE,
    SCAN_INTERVAL_SECONDS,
    WATCH_THRESHOLD,
)
from src.config.watchlist import WATCHLIST
from src.data.alpaca_data import AlpacaDataManager
from src.features.feature_engine import FeatureEngine
from src.journal.trade_journal import TradeJournal
from src.risk.risk_manager import RiskManager
from src.scanner.scanner import AIScanner


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"


def create_trading_client() -> TradingClient:
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


def refresh_symbol(
    symbol: str,
    data_manager: AlpacaDataManager,
    feature_engine: FeatureEngine,
) -> pd.DataFrame:
    print(f"Refreshing {symbol}...")

    dataframe = data_manager.download_5min(
        symbol=symbol,
        days=30,
    )

    if dataframe.empty:
        raise ValueError(
            f"No market data was returned for {symbol}."
        )

    dataframe.columns = [
        str(column).lower()
        for column in dataframe.columns
    ]

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{symbol} data is missing columns: "
            + ", ".join(missing_columns)
        )

    RAW_FOLDER.mkdir(parents=True, exist_ok=True)
    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

    raw_file = RAW_FOLDER / f"{symbol}_5min.csv"
    dataframe.to_csv(raw_file, index=False)

    features_dataframe = feature_engine.build(
        dataframe
    )

    processed_file = (
        PROCESSED_FOLDER
        / f"{symbol}_features.csv"
    )

    features_dataframe.to_csv(
        processed_file,
        index=False,
    )

    return features_dataframe


def display_rankings(results: list[dict]) -> None:
    results.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    print()
    print("=" * 65)
    print(" AI DAYTRADER RESULTS")
    print("=" * 65)
    print()
    print(
        f"{'Rank':<6}"
        f"{'Symbol':<9}"
        f"{'Confidence':<14}"
        f"Action"
    )
    print("-" * 48)

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
    print("=" * 65)


def display_trade_decision(
    decision: dict,
) -> None:
    print()
    print("=" * 65)
    print(" DRY-RUN TRADE DECISION")
    print("=" * 65)

    print(f"Symbol       : {decision['symbol']}")
    print(
        f"Confidence   : "
        f"{decision['confidence']:.2%}"
    )

    if not decision["approved"]:
        print("Decision     : REJECTED")

        for reason in decision["reasons"]:
            print(f"Reason       : {reason}")

        print("=" * 65)
        return

    print("Decision     : APPROVED")
    print(f"Shares       : {decision['shares']}")
    print(
        f"Entry        : "
        f"${decision['entry_price']:,.2f}"
    )
    print(
        f"Stop         : "
        f"${decision['stop_price']:,.2f}"
    )
    print(
        f"Target       : "
        f"${decision['target_price']:,.2f}"
    )
    print(
        f"Position     : "
        f"${decision['position_value']:,.2f}"
    )
    print(
        f"Maximum loss : "
        f"${decision['maximum_loss']:,.2f}"
    )
    print(
        f"Potential P/L: "
        f"${decision['potential_profit']:,.2f}"
    )
    print(
        f"Risk/reward  : "
        f"1:{decision['risk_reward_ratio']:.2f}"
    )

    print(
        "Order status : "
        "DRY RUN — no order submitted"
    )

    print("=" * 65)


def run_scan(
    trading_client: TradingClient,
    data_manager: AlpacaDataManager,
    feature_engine: FeatureEngine,
    scanner: AIScanner,
    risk_manager: RiskManager,
    journal: TradeJournal,
) -> None:
    print()
    print("=" * 65)
    print(" AI DAYTRADER")
    print("=" * 65)
    print(
        f"Scan started : "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )
    print()

    account = trading_client.get_account()
    positions = trading_client.get_all_positions()

    account_equity = float(account.equity)
    buying_power = float(account.buying_power)

    open_position_symbols = {
        position.symbol
        for position in positions
    }

    print(
        f"Account equity: "
        f"${account_equity:,.2f}"
    )
    print(
        f"Buying power  : "
        f"${buying_power:,.2f}"
    )
    print(
        f"Open positions: "
        f"{len(open_position_symbols)}"
    )
    print()

    results = []

    for symbol in WATCHLIST:
        try:
            features_dataframe = refresh_symbol(
                symbol=symbol,
                data_manager=data_manager,
                feature_engine=feature_engine,
            )

            confidence = scanner.score(
                features_dataframe
            )

            latest_complete_row = (
                features_dataframe
                .dropna()
                .iloc[-1]
            )

            entry_price = float(
                latest_complete_row["close"]
            )

            atr = float(
                latest_complete_row["ATR"]
            )

            action = get_action(confidence)

            result = {
                "symbol": symbol,
                "confidence": confidence,
                "action": action,
                "entry_price": entry_price,
                "atr": atr,
            }

            results.append(result)

            journal.log_scan(
                symbol=symbol,
                confidence=confidence,
                action=action,
                price=entry_price,
                atr=atr,
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
        print(
            "No symbols were successfully scanned."
        )
        return

    display_rankings(results)

    buy_candidates = [
        result
        for result in results
        if result["confidence"] >= BUY_THRESHOLD
    ]

    buy_candidates.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    if not buy_candidates:
        print()
        print(
            "No trades passed the buy threshold."
        )
        return

    available_position_slots = max(
        MAX_OPEN_POSITIONS
        - len(open_position_symbols),
        0,
    )

    if available_position_slots == 0:
        print()
        print(
            "No available position slots."
        )

    for candidate in buy_candidates:
        decision = risk_manager.evaluate_trade(
            symbol=candidate["symbol"],
            confidence=candidate["confidence"],
            minimum_confidence=BUY_THRESHOLD,
            entry_price=candidate["entry_price"],
            atr=candidate["atr"],
            account_equity=account_equity,
            buying_power=buying_power,
            open_position_symbols=(
                open_position_symbols
            ),
        )

        journal.log_trade_decision(
            decision=decision,
            dry_run=DRY_RUN,
        )

        display_trade_decision(decision)

        if (
            decision["approved"]
            and available_position_slots > 0
        ):
            buying_power -= decision[
                "position_value"
            ]

            open_position_symbols.add(
                decision["symbol"]
            )

            available_position_slots -= 1


def main() -> None:
    trading_client = create_trading_client()
    data_manager = AlpacaDataManager()
    feature_engine = FeatureEngine()
    scanner = AIScanner()
    journal = TradeJournal(PROJECT_ROOT)

    risk_manager = RiskManager(
        risk_per_trade=RISK_PER_TRADE,
        max_position_percent=(
            MAX_POSITION_PERCENT
        ),
        max_open_positions=(
            MAX_OPEN_POSITIONS
        ),
        atr_stop_multiplier=(
            ATR_STOP_MULTIPLIER
        ),
        atr_target_multiplier=(
            ATR_TARGET_MULTIPLIER
        ),
    )

    print()
    print(
        "AI-DayTrader started in "
        "DRY-RUN mode."
    )
    print("No orders will be submitted.")
    print("Press Ctrl+C to stop.")
    print()
    print(
        f"Scan journal: "
        f"{journal.scan_file}"
    )
    print(
        f"Trade journal: "
        f"{journal.trade_file}"
    )

    try:
        while True:
            clock = trading_client.get_clock()

            if clock.is_open:
                print()
                print("Market status: OPEN")
                print(
                    f"Next close: "
                    f"{clock.next_close}"
                )

                run_scan(
                    trading_client=trading_client,
                    data_manager=data_manager,
                    feature_engine=feature_engine,
                    scanner=scanner,
                    risk_manager=risk_manager,
                    journal=journal,
                )

                print(
                    f"\nNext scan in "
                    f"{SCAN_INTERVAL_SECONDS // 60} "
                    f"minutes."
                )

                time.sleep(
                    SCAN_INTERVAL_SECONDS
                )

            else:
                print()
                print("Market status: CLOSED")
                print(
                    f"Next open: "
                    f"{clock.next_open}"
                )
                print(
                    "Checking again in "
                    "5 minutes."
                )

                time.sleep(
                    SCAN_INTERVAL_SECONDS
                )

    except KeyboardInterrupt:
        print()
        print(
            "AI-DayTrader stopped safely."
        )


if __name__ == "__main__":
    main()