import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.alpaca_data import AlpacaDataManager
from src.config.watchlist import WATCHLIST

manager = AlpacaDataManager()

manager.download_watchlist(WATCHLIST)