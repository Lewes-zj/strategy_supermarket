# backend/tests/engine/test_portfolio.py
import pytest
from decimal import Decimal
from datetime import datetime

from engine.models import Order, OrderSide, OrderType, Fill
from engine.portfolio import Portfolio


class TestPortfolio:
    def test_initial_state(self):
        portfolio = Portfolio(cash=Decimal("1000000"))
        assert portfolio.cash == Decimal("1000000")
        assert len(portfolio.positions) == 0

    def test_get_position_creates_new(self):
        portfolio = Portfolio(cash=Decimal("1000000"))
        pos = portfolio.get_position("000001")
        assert pos.symbol == "000001"
        assert pos.quantity == Decimal("0")

    def test_process_buy_fill(self):
        portfolio = Portfolio(cash=Decimal("1000000"), initial_capital=Decimal("1000000"))
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
        portfolio.process_fill(fill)

        assert portfolio.cash == Decimal("1000000") - Decimal("1000") - Decimal("5")
        pos = portfolio.get_position("000001")
        assert pos.quantity == Decimal("100")
        assert pos.avg_cost == Decimal("10.00")

    def test_process_sell_fill(self):
        portfolio = Portfolio(cash=Decimal("999000"), initial_capital=Decimal("1000000"))
        # 先建立持仓
        portfolio.positions["000001"] = portfolio.get_position("000001")
        portfolio.positions["000001"].quantity = Decimal("100")
        portfolio.positions["000001"].avg_cost = Decimal("10.00")

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
        portfolio.process_fill(fill)

        # 卖出收入 1200 - 5 = 1195
        assert portfolio.cash == Decimal("999000") + Decimal("1200") - Decimal("5")
        pos = portfolio.get_position("000001")
        assert pos.quantity == Decimal("0")
        assert pos.realized_pnl == Decimal("200")  # (12-10)*100

    def test_get_equity(self):
        portfolio = Portfolio(cash=Decimal("900000"), initial_capital=Decimal("1000000"))
        portfolio.positions["000001"] = portfolio.get_position("000001")
        portfolio.positions["000001"].quantity = Decimal("100")

        prices = {"000001": Decimal("1000")}
        equity = portfolio.get_equity(prices)
        assert equity == Decimal("900000") + Decimal("100000")  # cash + 100*1000

    def test_get_weights(self):
        portfolio = Portfolio(cash=Decimal("500000"), initial_capital=Decimal("1000000"))
        portfolio.positions["000001"] = portfolio.get_position("000001")
        portfolio.positions["000001"].quantity = Decimal("100")
        portfolio.positions["600519"] = portfolio.get_position("600519")
        portfolio.positions["600519"].quantity = Decimal("50")

        prices = {
            "000001": Decimal("2500"),  # 100 * 2500 = 250000
            "600519": Decimal("5000"),  # 50 * 5000 = 250000
        }
        # 总权益 = 500000 + 250000 + 250000 = 1000000
        weights = portfolio.get_weights(prices)

        assert abs(weights["000001"] - 0.25) < 0.001
        assert abs(weights["600519"] - 0.25) < 0.001
