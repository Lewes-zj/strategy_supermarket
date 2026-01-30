# backend/tests/engine/test_backtester.py
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List

from engine.models import Order, OrderSide, OrderType, Fill, BacktestResult
from engine.backtester import EventDrivenBacktester, Strategy


class SimpleTestStrategy(Strategy):
    """简单测试策略：每10天买入，每20天卖出"""

    def __init__(self):
        self.day_count = 0
        self.position = 0

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        self.day_count += 1
        orders = []

        if self.day_count % 20 == 0 and self.position > 0:
            # 卖出
            orders.append(Order(
                symbol="000001",
                side=OrderSide.SELL,
                quantity=Decimal("100"),
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))
        elif self.day_count % 10 == 0 and self.position == 0:
            # 买入
            orders.append(Order(
                symbol="000001",
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))

        return orders

    def on_fill(self, fill: Fill) -> None:
        if fill.order.side == OrderSide.BUY:
            self.position += int(fill.fill_quantity)
        else:
            self.position -= int(fill.fill_quantity)


@pytest.fixture
def sample_data():
    """创建测试用多标的数据"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")

    # 000001 数据
    data_000001 = pd.DataFrame({
        "open": np.linspace(10, 12, 100) + np.random.normal(0, 0.1, 100),
        "high": np.linspace(10.2, 12.2, 100) + np.random.normal(0, 0.1, 100),
        "low": np.linspace(9.8, 11.8, 100) + np.random.normal(0, 0.1, 100),
        "close": np.linspace(10, 12, 100) + np.random.normal(0, 0.1, 100),
        "volume": np.random.randint(1000000, 2000000, 100)
    }, index=dates)

    return {"000001": data_000001}


@pytest.fixture
def benchmark_data():
    """创建基准数据"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
    return pd.DataFrame({
        "open": np.linspace(3000, 3200, 100),
        "high": np.linspace(3020, 3220, 100),
        "low": np.linspace(2980, 3180, 100),
        "close": np.linspace(3000, 3200, 100),
        "volume": np.random.randint(10000000, 20000000, 100)
    }, index=dates)


class TestEventDrivenBacktester:
    def test_backtest_runs(self, sample_data, benchmark_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(strategy)

        result = backtester.run(sample_data, benchmark_data)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert hasattr(result, 'metrics')

    def test_equity_curve_has_correct_columns(self, sample_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(strategy)

        result = backtester.run(sample_data)

        assert "equity" in result.equity_curve.columns
        assert "returns" in result.equity_curve.columns

    def test_initial_capital_is_respected(self, sample_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(
            strategy,
            initial_capital=Decimal("500000")
        )

        result = backtester.run(sample_data)

        # 第一天权益应该接近初始资金
        first_equity = result.equity_curve["equity"].iloc[0]
        assert abs(first_equity - 500000) < 1000

    def test_trades_are_recorded(self, sample_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(strategy)

        result = backtester.run(sample_data)

        # 100天内应该有多次交易
        assert len(result.trades) > 0

    def test_benchmark_comparison(self, sample_data, benchmark_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(strategy)

        result = backtester.run(sample_data, benchmark_data)

        # 应该有基准列
        assert "benchmark" in result.equity_curve.columns
        # Alpha和Beta应该已计算
        assert result.metrics.beta != 1.0 or result.metrics.alpha != 0.0
