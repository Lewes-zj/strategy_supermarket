# backend/tests/engine/test_execution.py
import pytest
import pandas as pd
from decimal import Decimal
from datetime import datetime

from engine.models import Order, OrderSide, OrderType
from engine.execution import (
    MarketExecutionModel, LimitExecutionModel,
    StopExecutionModel, CompositeExecutionModel
)


@pytest.fixture
def sample_bars():
    """创建测试用K线数据"""
    current_bar = pd.Series({
        "open": 10.00,
        "high": 10.50,
        "low": 9.80,
        "close": 10.20,
        "volume": 1000000
    }, name=datetime(2025, 1, 15, 15, 0, 0))

    next_bar = pd.Series({
        "open": 10.25,
        "high": 10.60,
        "low": 10.10,
        "close": 10.40,
        "volume": 1200000
    }, name=datetime(2025, 1, 16, 15, 0, 0))

    return current_bar, next_bar


class TestMarketExecutionModel:
    def test_buy_order_execution(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = MarketExecutionModel(slippage_bps=10, commission_rate=0.0003)

        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        # 买入用下一根K线开盘价 10.25 * (1 + 0.001) = 10.26025
        assert fill.fill_price > Decimal("10.25")
        assert fill.fill_quantity == Decimal("100")
        assert fill.commission >= Decimal("5")  # 最低佣金5元

    def test_sell_order_execution(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = MarketExecutionModel(slippage_bps=10, commission_rate=0.0003)

        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        # 卖出用下一根K线开盘价 10.25 / (1 + 0.001) = 10.23975
        assert fill.fill_price < Decimal("10.25")

    def test_no_next_bar_returns_none(self, sample_bars):
        current_bar, _ = sample_bars
        model = MarketExecutionModel()

        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )

        fill = model.execute(order, current_bar, None)
        assert fill is None


class TestLimitExecutionModel:
    def test_buy_limit_order_filled(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = LimitExecutionModel()

        # 限价10.15，下一根K线低点10.10，可以成交
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.15")
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        assert fill.fill_price == Decimal("10.15")

    def test_buy_limit_order_not_filled(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = LimitExecutionModel()

        # 限价10.05，下一根K线低点10.10，不能成交
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.05")
        )

        fill = model.execute(order, current_bar, next_bar)
        assert fill is None

    def test_sell_limit_order_filled(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = LimitExecutionModel()

        # 限价10.55，下一根K线高点10.60，可以成交
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.55")
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        assert fill.fill_price == Decimal("10.55")


class TestStopExecutionModel:
    def test_stop_loss_sell_triggered(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = StopExecutionModel(slippage_bps=20)

        # 止损价10.15，下一根K线低点10.10，触发止损
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.STOP,
            stop_price=Decimal("10.15")
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        # 止损触发后以开盘价成交，带滑点
        assert fill.fill_price < Decimal("10.25")

    def test_stop_loss_not_triggered(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = StopExecutionModel()

        # 止损价10.05，下一根K线低点10.10，不触发
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.STOP,
            stop_price=Decimal("10.05")
        )

        fill = model.execute(order, current_bar, next_bar)
        assert fill is None


class TestCompositeExecutionModel:
    def test_routes_market_order(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = CompositeExecutionModel()

        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )

        fill = model.execute(order, current_bar, next_bar)
        assert fill is not None

    def test_routes_limit_order(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = CompositeExecutionModel()

        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.15")
        )

        fill = model.execute(order, current_bar, next_bar)
        assert fill is not None
