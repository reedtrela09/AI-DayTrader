import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.strategy.ema_strategy import EMAStrategy
from src.backtest.backtester import Backtester

print("Loading features...")

df = pd.read_csv("data/processed/SPY_features.csv")

strategy = EMAStrategy()

df = strategy.generate_signals(df)

tester = Backtester()

tester.run(df)