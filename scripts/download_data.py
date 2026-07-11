import yfinance as yf
from pathlib import Path

# Create the data folder if it doesn't exist
data_folder = Path(__file__).parent.parent / "data"
data_folder.mkdir(exist_ok=True)

print("Downloading SPY data...")

spy = yf.download(
    "SPY",
    start="2020-01-01",
    progress=False
)

output_file = data_folder / "spy.csv"
spy.to_csv(output_file)

print(f"Done! Data saved to:\n{output_file}")