import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class AlpacaDataManager:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[2]

        load_dotenv(self.project_root / ".env")

        self.client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY")
        )

    def download_5min(self, symbol, days=30):

        end = datetime.now(timezone.utc) - timedelta(minutes=20)
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start,
            end=end,
        )

        bars = self.client.get_stock_bars(request).df

        bars = bars.reset_index()

        return bars

    def save_data(self, df, symbol):

        raw_folder = self.project_root / "data" / "raw"
        raw_folder.mkdir(parents=True, exist_ok=True)

        filename = raw_folder / f"{symbol}_5min.csv"

        df.to_csv(filename, index=False)

        print(f"✅ Saved {symbol} to {filename}")

    def download_watchlist(self, watchlist):

        for symbol in watchlist:

            print(f"Downloading {symbol}...")

            df = self.download_5min(symbol)

            self.save_data(df, symbol)

        print("\n✅ Finished downloading all symbols!")