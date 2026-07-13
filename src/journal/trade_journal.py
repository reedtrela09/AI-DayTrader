import csv
from datetime import datetime
from pathlib import Path


class TradeJournal:

    def __init__(self, project_root: Path):
        self.logs_folder = project_root / "logs"
        self.logs_folder.mkdir(parents=True, exist_ok=True)

        self.scan_file = self.logs_folder / "scan_journal.csv"
        self.trade_file = self.logs_folder / "trade_journal.csv"

        self.scan_columns = [
            "Timestamp",
            "Symbol",
            "Confidence",
            "Action",
            "Price",
            "ATR",
        ]

        self.trade_columns = [
            "Timestamp",
            "Symbol",
            "Confidence",
            "Approved",
            "Shares",
            "EntryPrice",
            "StopPrice",
            "TargetPrice",
            "PositionValue",
            "MaximumLoss",
            "PotentialProfit",
            "Reasons",
            "DryRun",
        ]

    @staticmethod
    def current_timestamp() -> str:
        return datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

    @staticmethod
    def append_row(
        file_path: Path,
        fieldnames: list[str],
        row: dict,
    ) -> None:
        file_exists = file_path.exists()

        with file_path.open(
            mode="a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def log_scan(
        self,
        symbol: str,
        confidence: float,
        action: str,
        price: float,
        atr: float,
    ) -> None:
        self.append_row(
            file_path=self.scan_file,
            fieldnames=self.scan_columns,
            row={
                "Timestamp": self.current_timestamp(),
                "Symbol": symbol,
                "Confidence": round(confidence, 6),
                "Action": action,
                "Price": round(price, 4),
                "ATR": round(atr, 4),
            },
        )

    def log_trade_decision(
        self,
        decision: dict,
        dry_run: bool,
    ) -> None:
        approved = bool(decision["approved"])

        reasons = "; ".join(
            decision.get("reasons", [])
        )

        self.append_row(
            file_path=self.trade_file,
            fieldnames=self.trade_columns,
            row={
                "Timestamp": self.current_timestamp(),
                "Symbol": decision["symbol"],
                "Confidence": round(
                    decision["confidence"],
                    6,
                ),
                "Approved": approved,
                "Shares": (
                    decision.get("shares", 0)
                    if approved
                    else 0
                ),
                "EntryPrice": (
                    round(
                        decision.get("entry_price", 0),
                        4,
                    )
                    if approved
                    else ""
                ),
                "StopPrice": (
                    round(
                        decision.get("stop_price", 0),
                        4,
                    )
                    if approved
                    else ""
                ),
                "TargetPrice": (
                    round(
                        decision.get("target_price", 0),
                        4,
                    )
                    if approved
                    else ""
                ),
                "PositionValue": (
                    round(
                        decision.get(
                            "position_value",
                            0,
                        ),
                        2,
                    )
                    if approved
                    else ""
                ),
                "MaximumLoss": (
                    round(
                        decision.get(
                            "maximum_loss",
                            0,
                        ),
                        2,
                    )
                    if approved
                    else ""
                ),
                "PotentialProfit": (
                    round(
                        decision.get(
                            "potential_profit",
                            0,
                        ),
                        2,
                    )
                    if approved
                    else ""
                ),
                "Reasons": reasons,
                "DryRun": dry_run,
            },
        )