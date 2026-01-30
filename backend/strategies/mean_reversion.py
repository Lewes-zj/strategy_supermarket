"""
Mean Reversion Strategy using RSI and Bollinger Bands.
Designed for CSI 300 constituents - identifies oversold conditions for potential reversals.
"""
from datetime import datetime
from decimal import Decimal
from typing import List
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from engine.backtester import Strategy
from engine.models import Order, OrderSide, OrderType, Fill


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy for CSI 300 stocks.

    Logic:
    - Entry: RSI < 30 (oversold) AND price below lower Bollinger Band
    - Exit: RSI > 70 (overbought) OR price crosses above middle band
    - Stop Loss: 5% below entry price

    Suitable for:
    - Range-bound markets
    - Short-term trading (3-10 day holding period)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_period: int = 20,
        bb_std: float = 2,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stop_loss_pct = Decimal(str(stop_loss_pct))
        self.take_profit_pct = Decimal(str(take_profit_pct))

        # Position tracking
        self.entry_price = Decimal("0")
        self.entry_date = None

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """Generate trading signals based on mean reversion indicators."""
        orders = []

        if len(data) < max(self.rsi_period, self.bb_period) + 1:
            return orders

        # Calculate indicators
        indicators = self._calculate_indicators(data)

        if indicators is None:
            return orders

        # Get current values
        current_rsi = indicators['rsi'].iloc[-1]
        current_price = Decimal(str(data['close'].iloc[-1]))
        lower_band = indicators['lower_band'].iloc[-1]
        upper_band = indicators['upper_band'].iloc[-1]
        middle_band = indicators['middle_band'].iloc[-1]

        symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else "UNKNOWN"

        # Check if we have a position (entry_price > 0)
        has_position = self.entry_price > 0

        if has_position:
            # Exit conditions
            should_exit = False
            exit_reason = ""

            # Check stop loss
            stop_price = self.entry_price * (Decimal("1") - self.stop_loss_pct)
            if current_price < stop_price:
                should_exit = True
                exit_reason = "stop_loss"

            # Check take profit
            profit_price = self.entry_price * (Decimal("1") + self.take_profit_pct)
            if current_price > profit_price:
                should_exit = True
                exit_reason = "take_profit"

            # Check RSI overbought
            if current_rsi > self.rsi_overbought:
                should_exit = True
                exit_reason = "rsi_overbought"

            # Check price above middle band
            if current_price > Decimal(str(middle_band)):
                should_exit = True
                exit_reason = "above_middle_band"

            if should_exit:
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=Decimal("100"),
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                ))
                # Note: Position cleared in on_fill

        else:
            # Entry conditions: Oversold + below lower band
            if current_rsi < self.rsi_oversold and current_price < Decimal(str(lower_band)):
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=Decimal("100"),
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                ))

        return orders

    def on_fill(self, fill: Fill) -> None:
        """Handle order fills and update position state."""
        if fill.order.side == OrderSide.BUY:
            self.entry_price = fill.fill_price
            self.entry_date = fill.timestamp
        elif fill.order.side == OrderSide.SELL:
            self.entry_price = Decimal("0")
            self.entry_date = None

    def _calculate_indicators(self, data: pd.DataFrame) -> dict:
        """Calculate RSI and Bollinger Bands."""
        try:
            closes = data['close']

            # Calculate RSI
            rsi = self._calculate_rsi(closes, self.rsi_period)

            # Calculate Bollinger Bands
            bb = self._calculate_bollinger_bands(closes, self.bb_period, self.bb_std)

            return {
                'rsi': rsi,
                'upper_band': bb['upper'],
                'middle_band': bb['middle'],
                'lower_band': bb['lower']
            }

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int,
        std_dev: float
    ) -> dict:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }


class MultiStockMeanReversion(MeanReversionStrategy):
    """
    Extended mean reversion strategy for multiple stocks.
    Selects the most oversold stock from a watchlist.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.symbols_scores = {}  # Track oversold scores per symbol

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """
        Generate signals for multiple stocks.
        Returns orders for the most oversold stock if conditions are met.
        """
        orders = []

        # Calculate oversold score for this symbol
        if len(data) >= self.rsi_period + 1:
            indicators = self._calculate_indicators(data)
            if indicators:
                current_rsi = indicators['rsi'].iloc[-1]
                symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else "UNKNOWN"

                # Lower RSI = higher score (more oversold)
                self.symbols_scores[symbol] = 100 - current_rsi

        # Use parent logic for actual signal generation
        return super().on_bar(timestamp, data)

    def get_most_oversold(self) -> str:
        """Get the most oversold symbol from tracked scores."""
        if not self.symbols_scores:
            return None

        return max(self.symbols_scores, key=self.symbols_scores.get)
