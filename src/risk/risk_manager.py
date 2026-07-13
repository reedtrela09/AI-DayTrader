import math


class RiskManager:
    def __init__(
        self,
        risk_per_trade: float,
        max_position_percent: float,
        max_open_positions: int,
        atr_stop_multiplier: float,
        atr_target_multiplier: float,
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_percent = max_position_percent
        self.max_open_positions = max_open_positions
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_target_multiplier = atr_target_multiplier

    def evaluate_trade(
        self,
        symbol: str,
        confidence: float,
        minimum_confidence: float,
        entry_price: float,
        atr: float,
        account_equity: float,
        buying_power: float,
        open_position_symbols: set[str],
    ) -> dict:
        rejection_reasons = []

        if confidence < minimum_confidence:
            rejection_reasons.append(
                "Confidence is below the buy threshold."
            )

        if symbol in open_position_symbols:
            rejection_reasons.append(
                f"An open {symbol} position already exists."
            )

        if len(open_position_symbols) >= self.max_open_positions:
            rejection_reasons.append(
                "Maximum number of open positions has been reached."
            )

        if entry_price <= 0:
            rejection_reasons.append("Entry price is invalid.")

        if atr <= 0:
            rejection_reasons.append("ATR is invalid.")

        if rejection_reasons:
            return {
                "approved": False,
                "symbol": symbol,
                "confidence": confidence,
                "reasons": rejection_reasons,
            }

        stop_distance = atr * self.atr_stop_multiplier
        target_distance = atr * self.atr_target_multiplier

        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance

        risk_budget = account_equity * self.risk_per_trade
        maximum_position_value = (
            account_equity * self.max_position_percent
        )

        shares_by_risk = math.floor(
            risk_budget / stop_distance
        )

        shares_by_position_limit = math.floor(
            maximum_position_value / entry_price
        )

        shares_by_buying_power = math.floor(
            buying_power / entry_price
        )

        shares = min(
            shares_by_risk,
            shares_by_position_limit,
            shares_by_buying_power,
        )

        if shares < 1:
            return {
                "approved": False,
                "symbol": symbol,
                "confidence": confidence,
                "reasons": [
                    "Position size calculated to fewer than one share."
                ],
            }

        position_value = shares * entry_price
        maximum_loss = shares * stop_distance
        potential_profit = shares * target_distance

        return {
            "approved": True,
            "symbol": symbol,
            "confidence": confidence,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "shares": shares,
            "position_value": position_value,
            "maximum_loss": maximum_loss,
            "potential_profit": potential_profit,
            "risk_reward_ratio": (
                target_distance / stop_distance
            ),
            "reasons": [],
        }