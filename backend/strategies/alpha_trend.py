from datetime import datetime
from decimal import Decimal
from typing import List
import pandas as pd
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.backtester import Strategy, Order, OrderSide, OrderType, Fill

class AlphaTrendStrategy(Strategy):
    """
    Alpha Trend Strategy:
    - Trend following using SMA Crossover.
    - Risk management using fixed percentage Stop Loss.
    """
    def __init__(self, short_window: int = 10, long_window: int = 30, stop_loss_pct: float = 0.05):
        self.short_window = short_window
        self.long_window = long_window
        self.stop_loss_pct = Decimal(str(stop_loss_pct))
        
        self.position_entry_price = Decimal("0")
        self.position_held = False # Simple boolean (assuming single asset per instance for now)

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        orders = []
        if len(data) < self.long_window:
            return orders

        # Calculate indicators
        closes = data["close"]
        short_ma = closes.rolling(window=self.short_window).mean()
        long_ma = closes.rolling(window=self.long_window).mean()

        curr_short = short_ma.iloc[-1]
        curr_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        current_price = Decimal(str(closes.iloc[-1]))
        symbol = data["symbol"].iloc[-1] if "symbol" in data.columns else "UNKNOWN"

        # Trading Logic
        
        # Check Stop Loss if holding
        if self.position_held:
            stop_price = self.position_entry_price * (Decimal("1") - self.stop_loss_pct)
            if current_price < stop_price:
                # Stop Loss Triggered
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=Decimal("100"), # Fixed qty for simplicity
                    order_type=OrderType.MARKET
                ))
                self.position_held = False
                return orders

        # Entry Signal: Golden Cross
        if not self.position_held and prev_short <= prev_long and curr_short > curr_long:
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.MARKET
            ))
            # Note: position_held set to True in on_fill usually, but for simple logic we can track intent
            # Correct way is to wait for fill, but simple strategy might assume fill. 
            # We will handle state in on_fill.

        # Exit Signal: Death Cross
        elif self.position_held and prev_short >= prev_long and curr_short < curr_long:
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=Decimal("100"),
                order_type=OrderType.MARKET
            ))

        return orders

    def on_fill(self, fill: Fill) -> None:
        if fill.order.side == OrderSide.BUY:
            self.position_held = True
            self.position_entry_price = fill.fill_price
        elif fill.order.side == OrderSide.SELL:
            self.position_held = False
            self.position_entry_price = Decimal("0")
