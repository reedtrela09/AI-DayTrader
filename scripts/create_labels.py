import pandas as pd
from pathlib import Path

project_folder = Path(__file__).parent.parent
data_folder = project_folder / "data"

print("Loading feature data...")

df = pd.read_csv(data_folder / "features.csv")

LOOKAHEAD = 10        # Look 10 trading days into the future
PROFIT_TARGET = 0.02  # 2% gain

future_price = df["Close"].shift(-LOOKAHEAD)

future_return = (future_price - df["Close"]) / df["Close"]

df["Target"] = (future_return >= PROFIT_TARGET).astype(int)

output = data_folder / "training_data.csv"

df.to_csv(output, index=False)

print("\nDone!")
print(df[["Date", "Close", "Target"]].tail(15))