import pandas as pd


class Backtester:

    def __init__(self):

        self.start_balance = 100000

        self.balance = self.start_balance

        self.position = 0

        self.entry = 0

        self.trade_log = []

    def buy(self, price):

        self.position = 1

        self.entry = price

    def sell(self, price):

        pnl = price - self.entry

        self.balance += pnl

        self.trade_log.append(
            {
                "Entry": self.entry,
                "Exit": price,
                "PnL": pnl,
                "Balance": self.balance
            }
        )

        self.position = 0

    def run(self, df):

        for _, row in df.iterrows():

            signal = row["Signal"]

            price = row["close"]

            if signal == 1 and self.position == 0:

                self.buy(price)

            elif signal == -1 and self.position == 1:

                self.sell(price)

        self.report()

    def report(self):

        trades = pd.DataFrame(self.trade_log)

        if trades.empty:

            print("No trades.")

            return

        wins = trades[trades["PnL"] > 0]

        losses = trades[trades["PnL"] <= 0]

        print()

        print("=" * 40)

        print("BACKTEST REPORT")

        print("=" * 40)

        print(f"Starting Balance : ${self.start_balance:,.2f}")

        print(f"Ending Balance   : ${self.balance:,.2f}")

        print(f"Net Profit       : ${self.balance-self.start_balance:,.2f}")

        print(f"Trades           : {len(trades)}")

        print(f"Wins             : {len(wins)}")

        print(f"Losses           : {len(losses)}")

        print(f"Win Rate         : {len(wins)/len(trades):.2%}")

        print(f"Average Win      : ${wins['PnL'].mean():.2f}")

        print(f"Average Loss     : ${losses['PnL'].mean():.2f}")

        print()

        print(trades.tail())