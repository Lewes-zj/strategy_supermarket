"""
Sector Rotation Strategy for CSI 300 sectors.
Rotates capital between sectors based on relative strength and momentum.
"""
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from engine.backtester import Strategy
from engine.models import Order, OrderSide, OrderType, Fill


class SectorRotationStrategy(Strategy):
    """
    Sector Rotation Strategy using CSI 300 sector indices/ETFs.

    Logic:
    - Calculate relative strength score for each sector
    - Allocate to top N performing sectors
    - Rebalance when relative strength changes significantly

    Suitable for:
    - Sector-level trend following
    - Quarterly/Monthly rebalancing
    - Risk diversification across sectors
    """

    # CSI 300 Sector mapping (simplified)
    SECTORS = {
        'finance': ['金融', '银行', '证券', '保险'],
        'technology': ['科技', '电子', '计算机', '通信'],
        'healthcare': ['医药', '生物', '医疗'],
        'consumer': ['消费', '食品', '饮料', '零售'],
        'energy': ['能源', '电力', '煤炭', '石油'],
        'materials': ['材料', '化工', '钢铁', '有色'],
        'industrial': ['制造', '机械', '军工', '建筑'],
        'utilities': ['公用', '环保', '水务']
    }

    def __init__(
        self,
        lookback_period: int = 20,
        top_n_sectors: int = 3,
        rebalance_threshold: float = 0.10,
        stop_loss_pct: float = 0.05
    ):
        self.lookback_period = lookback_period
        self.top_n_sectors = top_n_sectors
        self.rebalance_threshold = rebalance_threshold
        self.stop_loss_pct = Decimal(str(stop_loss_pct))

        # Position tracking per sector
        self.sector_positions: Dict[str, Dict] = {}
        self.current_allocations: Dict[str, Decimal] = {}

        # Momentum scores for each sector
        self.sector_scores: Dict[str, float] = {}

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """Generate rotation signals based on sector momentum."""
        orders = []

        if len(data) < self.lookback_period + 1:
            return orders

        # Get symbol/sector info
        symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else "UNKNOWN"

        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(data)
        self.sector_scores[symbol] = momentum_score

        # Check if we need to rebalance (simplified - checking this symbol)
        if symbol in self.sector_positions:
            position = self.sector_positions[symbol]

            # Check stop loss
            current_price = Decimal(str(data['close'].iloc[-1]))
            if position['entry_price'] > 0:
                stop_price = position['entry_price'] * (Decimal("1") - self.stop_loss_pct)
                if current_price < stop_price:
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=position['quantity'],
                        order_type=OrderType.MARKET,
                        timestamp=timestamp
                    ))
                    return orders

        # For simplicity, this is a placeholder for sector rotation logic
        # Full implementation would track all sectors and rebalance accordingly

        return orders

    def on_fill(self, fill: Fill) -> None:
        """Track sector positions."""
        symbol = fill.order.symbol

        if fill.order.side == OrderSide.BUY:
            self.sector_positions[symbol] = {
                'entry_price': fill.fill_price,
                'quantity': fill.fill_quantity,
                'entry_date': fill.timestamp
            }
        elif fill.order.side == OrderSide.SELL:
            if symbol in self.sector_positions:
                del self.sector_positions[symbol]

    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score for a sector/symbol."""
        closes = data['close']

        # Calculate returns over different periods
        ret_5d = (closes.iloc[-1] / closes.iloc[-min(6, len(closes))] - 1) * 100 if len(closes) > 5 else 0
        ret_20d = (closes.iloc[-1] / closes.iloc[-min(21, len(closes))] - 1) * 100 if len(closes) > 20 else 0
        ret_60d = (closes.iloc[-1] / closes.iloc[-min(61, len(closes))] - 1) * 100 if len(closes) > 60 else 0

        # Composite score (weighted average)
        score = (ret_5d * 0.2 + ret_20d * 0.5 + ret_60d * 0.3)

        return score

    def get_top_sectors(self, n: int = None) -> List[str]:
        """Get top N sectors by momentum score."""
        if n is None:
            n = self.top_n_sectors

        sorted_sectors = sorted(
            self.sector_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [s[0] for s in sorted_sectors[:n]]

    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        # Simple rebalance logic based on score changes
        # In production, would use more sophisticated logic
        return len(self.sector_positions) != self.top_n_sectors


class RelativeStrengthSectorRotation(SectorRotationStrategy):
    """
    Enhanced sector rotation using relative strength vs benchmark.
    """

    def __init__(self, benchmark_symbol: str = "000300", **kwargs):
        super().__init__(**kwargs)
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_data: pd.DataFrame = pd.DataFrame()

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """Generate signals based on relative strength vs benchmark."""
        orders = []

        if len(data) < self.lookback_period + 1:
            return orders

        symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else "UNKNOWN"

        # Calculate relative strength
        rs_score = self._calculate_relative_strength(data)

        # Entry: Strong relative strength (RS > 100)
        if rs_score > 100 and symbol not in self.sector_positions:
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))

        # Exit: Weak relative strength (RS < 95)
        elif rs_score < 95 and symbol in self.sector_positions:
            position = self.sector_positions[symbol]
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position['quantity'],
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))

        return orders

    def _calculate_relative_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate relative strength vs benchmark.

        RS = (Security Return / Benchmark Return) * 100
        """
        closes = data['close']

        # Calculate security return
        if len(closes) < self.lookback_period:
            return 100

        sec_return = (closes.iloc[-1] / closes.iloc[-self.lookback_period] - 1) * 100

        # For this simplified version, assume benchmark return is 5%
        # In production, would fetch actual benchmark data
        benchmark_return = 5.0

        # Calculate RS
        if benchmark_return != 0:
            rs = ((sec_return / 100 + 1) / (benchmark_return / 100 + 1)) * 100
        else:
            rs = 100

        return rs


class MomentumSectorRotation(SectorRotationStrategy):
    """
    Sector rotation using pure momentum ranking.
    """

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """Generate signals based on momentum ranking."""
        orders = []

        if len(data) < self.lookback_period + 1:
            return orders

        symbol = data['symbol'].iloc[-1] if 'symbol' in data.columns else "UNKNOWN"

        # Calculate momentum score
        momentum = self._calculate_momentum_score(data)

        # Update score
        self.sector_scores[symbol] = momentum

        # Check if this symbol is in top N and should be held
        top_sectors = self.get_top_sectors()

        if symbol in top_sectors and symbol not in self.sector_positions:
            # Entry signal - in top N and not held
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))

        elif symbol not in top_sectors and symbol in self.sector_positions:
            # Exit signal - not in top N but currently held
            position = self.sector_positions[symbol]
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position['quantity'],
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))

        return orders
