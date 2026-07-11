import pandas as pd


class EMAStrategy:

    def generate_signals(self, df: pd.DataFrame):

        df = df.copy()

        df["Signal"] = 0

        buy = (
            (df["EMA9"] > df["EMA21"]) &
            (df["EMA9"].shift(1) <= df["EMA21"].shift(1))
        )

        sell = (
            (df["EMA9"] < df["EMA21"]) &
            (df["EMA9"].shift(1) >= df["EMA21"].shift(1))
        )

        df.loc[buy, "Signal"] = 1
        df.loc[sell, "Signal"] = -1

        return df