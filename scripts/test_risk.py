import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.risk.risk_manager import RiskManager

risk = RiskManager(
    account_size=100000,
    risk_percent=1,
    max_position_percent=20
)

shares = risk.calculate_position_size(
    entry_price=750,
    stop_price=747
)

print()

print("Shares:", shares)
print("Capital:", shares * 750)
print("Maximum Risk:", shares * 3)