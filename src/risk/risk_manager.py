import math


class RiskManager:

    def __init__(
        self,
        account_size=100000,
        risk_percent=1.0,
        max_position_percent=20
    ):

        self.account_size = account_size
        self.risk_percent = risk_percent
        self.max_position_percent = max_position_percent

    def calculate_position_size(
        self,
        entry_price,
        stop_price
    ):

        risk_amount = self.account_size * (self.risk_percent / 100)

        risk_per_share = abs(entry_price - stop_price)

        if risk_per_share <= 0:
            return 0

        # Shares based on risk
        risk_shares = risk_amount / risk_per_share

        # Maximum capital allowed in one position
        max_position_value = (
            self.account_size *
            (self.max_position_percent / 100)
        )

        # Shares based on available capital
        capital_shares = max_position_value / entry_price

        shares = min(risk_shares, capital_shares)

        return math.floor(shares)