"""
Momentum Strategy for CSI 300 stocks.
Uses ROC (Rate of Change) and ADX (Average Directional Index) to identify trends.
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


class MomentumStrategy(Strategy):
    """
    Momentum Strategy for CSI 300 stocks.

    Logic:
    - Entry: ROC > threshold (positive momentum) AND ADX > 25 (strong trend)
    - Exit: ROC turns negative OR ADX drops below 20 (weakening trend)
    - Stop Loss: 8% below entry price (wider due to trend volatility)

    Suitable for:
    - Trending markets
    - Mid-term trading (1-4 week holding period)
    """

    def __init__(
        self,
        roc_period: int = 20,
        roc_threshold: float = 3,
        adx_period: int = 14,
        adx_trend_threshold: float = 25,
        adx_weak_threshold: float = 20,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.20
    ):
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_weak_threshold = adx_weak_threshold
        self.stop_loss_pct = Decimal(str(stop_loss_pct))
        self.take_profit_pct = Decimal(str(take_profit_pct))

        # Position tracking
        self.entry_price = Decimal("0")
        self.entry_date = None
        self.highest_price = Decimal("0")  # For trailing stop

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """Generate trading signals based on momentum indicators."""
        orders = []

        if len(data) < self.roc_period + self.adx_period + 1:
            return orders

        # Calculate indicators
        indicators = self._calculate_indicators(data)

        if indicators is None or indicators['roc'].isna().iloc[-1]:
            return orders

        # Get current values
        current_roc = indicators['roc'].iloc[-1]
        current_adx = indicators['adx'].iloc[-1]
        current_plus_di = indicators['plus_di'].iloc[-1]
        current_minus_di = indicators['minus_di'].iloc[-1]

        current_price = Decimal(str(data['close'].iloc[-1]))

        symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else "UNKNOWN"

        # Check if we have a position
        has_position = self.entry_price > 0

        if has_position:
            # Update highest price for trailing stop
            if current_price > self.highest_price:
                self.highest_price = current_price

            # Exit conditions
            should_exit = False
            exit_reason = ""

            # Check stop loss (fixed)
            stop_price = self.entry_price * (Decimal("1") - self.stop_loss_pct)
            if current_price < stop_price:
                should_exit = True
                exit_reason = "stop_loss"

            # Check trailing stop (3% below highest)
            trailing_stop = self.highest_price * Decimal("0.97")
            if current_price < trailing_stop:
                should_exit = True
                exit_reason = "trailing_stop"

            # Check take profit
            profit_price = self.entry_price * (Decimal("1") + self.take_profit_pct)
            if current_price > profit_price:
                should_exit = True
                exit_reason = "take_profit"

            # Check momentum reversal (ROC turns negative)
            if current_roc < 0:
                should_exit = True
                exit_reason = "momentum_reversal"

            # Check weakening trend
            if current_adx < self.adx_weak_threshold:
                should_exit = True
                exit_reason = "weak_trend"

            # Check bearish crossover (minus DI crosses above plus DI)
            if current_minus_di > current_plus_di:
                should_exit = True
                exit_reason = "bearish_crossover"

            if should_exit:
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=Decimal("100"),
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                ))

        else:
            # Entry conditions: Positive ROC + Strong ADX trend + Bullish DI
            if (current_roc > self.roc_threshold and
                current_adx > self.adx_trend_threshold and
                current_plus_di > current_minus_di):
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
            self.highest_price = fill.fill_price
        elif fill.order.side == OrderSide.SELL:
            self.entry_price = Decimal("0")
            self.entry_date = None
            self.highest_price = Decimal("0")

    def _calculate_indicators(self, data: pd.DataFrame) -> dict:
        """Calculate ROC, ADX, and DI indicators."""
        try:
            highs = data['high']
            lows = data['low']
            closes = data['close']

            # Calculate ROC (Rate of Change)
            roc = ((closes - closes.shift(self.roc_period)) / closes.shift(self.roc_period)) * 100

            # Calculate ADX and DI
            adx, plus_di, minus_di = self._calculate_adx(
                highs, lows, closes, self.adx_period
            )

            return {
                'roc': roc,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None

    def _calculate_adx(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        period: int
    ):
        """Calculate Average Directional Index and Directional Indicators."""
        # Calculate True Range
        tr1 = highs - lows
        tr2 = abs(highs - closes.shift())
        tr3 = abs(lows - closes.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM
        up_move = highs - highs.shift()
        down_move = lows.shift() - lows

        plus_dm = pd.Series(index=highs.index, dtype=float)
        minus_dm = pd.Series(index=highs.index, dtype=float)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Smooth
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di


class DualMomentumStrategy(MomentumStrategy):
    """
    Dual Momentum Strategy combining absolute and relative momentum.

    Uses:
    - Absolute momentum: Price above 200-day MA
    - Relative momentum: Best performing among CSI 300 sectors
    """

    def __init__(self, ma_period: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.ma_period = ma_period

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """Generate signals with additional MA filter."""
        orders = []

        if len(data) < self.ma_period:
            return orders

        # Calculate 200-day MA
        closes = data['close']
        ma_200 = closes.rolling(window=self.ma_period).mean()
        current_price = closes.iloc[-1]
        current_ma = ma_200.iloc[-1]

        # Only consider long positions if price is above MA (absolute momentum)
        if current_price < current_ma:
            # Exit any existing position
            if self.entry_price > 0:
                symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else "UNKNOWN"
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=Decimal("100"),
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                ))
            return orders

        # Use parent momentum logic for entry/exit
        return super().on_bar(timestamp, data)
