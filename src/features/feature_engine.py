import pandas as pd

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice


class FeatureEngine:

    def build(self, df):

        df = df.copy()

        print("Calculating indicators...")

        # ==========================
        # Trend
        # ==========================

        df["EMA9"] = EMAIndicator(df["close"], window=9).ema_indicator()
        df["EMA21"] = EMAIndicator(df["close"], window=21).ema_indicator()
        df["EMA50"] = EMAIndicator(df["close"], window=50).ema_indicator()
        df["EMA200"] = EMAIndicator(df["close"], window=200).ema_indicator()

        df["EMA9_EMA21"] = df["EMA9"] - df["EMA21"]
        df["EMA21_EMA50"] = df["EMA21"] - df["EMA50"]

        df["DistEMA200"] = (
            (df["close"] - df["EMA200"]) / df["EMA200"]
        )

        # ==========================
        # Momentum
        # ==========================

        df["RSI"] = RSIIndicator(df["close"], window=14).rsi()

        macd = MACD(df["close"])

        df["MACD"] = macd.macd()
        df["MACDSignal"] = macd.macd_signal()
        df["MACDHist"] = macd.macd_diff()

        df["ROC"] = ROCIndicator(
            close=df["close"],
            window=10
        ).roc()

        # ==========================
        # Volatility
        # ==========================

        atr = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14
        )

        df["ATR"] = atr.average_true_range()
        df["ATRPercent"] = df["ATR"] / df["close"]

        bb = BollingerBands(
            close=df["close"],
            window=20
        )

        df["BBUpper"] = bb.bollinger_hband()
        df["BBLower"] = bb.bollinger_lband()
        df["BBWidth"] = (
            (df["BBUpper"] - df["BBLower"])
            / df["close"]
        )

        # ==========================
        # Volume
        # ==========================

        df["VolumeSMA20"] = (
            df["volume"]
            .rolling(20)
            .mean()
        )

        df["RelativeVolume"] = (
            df["volume"] / df["VolumeSMA20"]
        )

        try:
            vwap = VolumeWeightedAveragePrice(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                window=14
            )

            df["VWAP"] = vwap.volume_weighted_average_price()
            df["VWAPDistance"] = (
                (df["close"] - df["VWAP"])
                / df["VWAP"]
            )

        except Exception:
            pass

        # ==========================
        # Candlestick Features
        # ==========================

        df["Body"] = (
            (df["close"] - df["open"]).abs()
        )

        df["UpperWick"] = (
            df["high"] - df[["open", "close"]].max(axis=1)
        )

        df["LowerWick"] = (
            df[["open", "close"]].min(axis=1)
            - df["low"]
        )

        df["BodyPercent"] = (
            df["Body"] / df["close"]
        )

        # ==========================
        # Time Features
        # ==========================

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["Hour"] = df["timestamp"].dt.hour
        df["Minute"] = df["timestamp"].dt.minute

        # ==========================
        # Returns
        # ==========================

        df["Return"] = df["close"].pct_change()

        return df