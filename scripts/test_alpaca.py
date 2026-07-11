import os
from pathlib import Path

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load the .env file
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("API keys were not found in the .env file.")

client = TradingClient(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
    paper=True
)

account = client.get_account()

print("✅ Connected to Alpaca!")
print(f"Status: {account.status}")
print(f"Buying Power: ${account.buying_power}")
print(f"Cash: ${account.cash}")