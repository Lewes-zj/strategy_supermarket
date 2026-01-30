# backend/tests/engine/test_models.py
import pytest
from decimal import Decimal
from datetime import datetime

from engine.models import (
    OrderSide, OrderType, Order, Fill, Position,
    Trade, BacktestResult, PerformanceMetrics
)


class TestOrderSide:
    def test_buy_value(self):
        assert OrderSide.BUY.value == "buy"

    def test_sell_value(self):
        assert OrderSide.SELL.value == "sell"


class TestOrderType:
    def test_market_value(self):
        assert OrderType.MARKET.value == "market"

    def test_limit_value(self):
        assert OrderType.LIMIT.value == "limit"

    def test_stop_value(self):
        assert OrderType.STOP.value == "stop"


class TestOrder:
    def test_market_order_creation(self):
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        assert order.symbol == "000001"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("100")
        assert order.order_type == OrderType.MARKET
        assert order.limit_price is None
        assert order.stop_price is None

    def test_limit_order_creation(self):
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.50")
        )
        assert order.limit_price == Decimal("10.50")

    def test_stop_order_creation(self):
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.STOP,
            stop_price=Decimal("9.50")
        )
        assert order.stop_price == Decimal("9.50")


class TestFill:
    def test_fill_creation(self):
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        fill = Fill(
            order=order,
            fill_price=Decimal("10.55"),
            fill_quantity=Decimal("100"),
            commission=Decimal("5.00"),
            slippage=Decimal("0.50"),
            timestamp=datetime(2025, 1, 15, 9, 30, 0)
        )
        assert fill.fill_price == Decimal("10.55")
        assert fill.fill_quantity == Decimal("100")
        assert fill.commission == Decimal("5.00")


class TestPosition:
    def test_position_initial_state(self):
        pos = Position(symbol="000001")
        assert pos.symbol == "000001"
        assert pos.quantity == Decimal("0")
        assert pos.avg_cost == Decimal("0")
        assert pos.realized_pnl == Decimal("0")

    def test_position_update_buy(self):
        pos = Position(symbol="000001")
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        fill = Fill(
            order=order,
            fill_price=Decimal("10.00"),
            fill_quantity=Decimal("100"),
            commission=Decimal("5.00"),
            slippage=Decimal("0"),
            timestamp=datetime.now()
        )
        pos.update(fill)
        assert pos.quantity == Decimal("100")
        assert pos.avg_cost == Decimal("10.00")

    def test_position_update_sell_with_profit(self):
        pos = Position(symbol="000001", quantity=Decimal("100"), avg_cost=Decimal("10.00"))
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        fill = Fill(
            order=order,
            fill_price=Decimal("12.00"),
            fill_quantity=Decimal("100"),
            commission=Decimal("5.00"),
            slippage=Decimal("0"),
            timestamp=datetime.now()
        )
        pos.update(fill)
        assert pos.quantity == Decimal("0")
        assert pos.realized_pnl == Decimal("200.00")  # (12-10) * 100
