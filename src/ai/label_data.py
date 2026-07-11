import pandas as pd


class LabelGenerator:

    def generate(
        self,
        df: pd.DataFrame,
        atr_tp=1.5,
        atr_sl=1.0,
        lookahead=12
    ):

        df = df.copy()

        labels = []

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atr = df["ATR"].values

        for i in range(len(df)):

            if i + lookahead >= len(df):
                labels.append(None)
                continue

            entry = closes[i]

            take_profit = entry + atr[i] * atr_tp
            stop_loss = entry - atr[i] * atr_sl

            label = 0

            for j in range(i + 1, i + lookahead + 1):

                if highs[j] >= take_profit:
                    label = 1
                    break

                if lows[j] <= stop_loss:
                    label = 0
                    break

            labels.append(label)

        df["Target"] = labels

        return df