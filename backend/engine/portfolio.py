# backend/engine/portfolio.py
"""
组合管理模块：支持多标的持仓管理
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List

from .models import Position, Fill, Order, OrderSide


@dataclass
class Portfolio:
    """投资组合管理器"""
    cash: Decimal
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_capital: Decimal = Decimal("0")

    def __post_init__(self):
        if self.initial_capital == Decimal("0"):
            self.initial_capital = self.cash

    def get_position(self, symbol: str) -> Position:
        """获取指定标的持仓，不存在则创建"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def process_fill(self, fill: Fill) -> None:
        """处理成交"""
        position = self.get_position(fill.order.symbol)
        position.update(fill)

        if fill.order.side == OrderSide.BUY:
            self.cash -= fill.fill_price * fill.fill_quantity + fill.commission
        else:
            self.cash += fill.fill_price * fill.fill_quantity - fill.commission

    def get_equity(self, prices: Dict[str, Decimal]) -> Decimal:
        """计算组合总权益"""
        equity = self.cash
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in prices:
                equity += position.quantity * prices[symbol]
        return equity

    def get_weights(self, prices: Dict[str, Decimal]) -> Dict[str, float]:
        """计算各标的权重"""
        equity = self.get_equity(prices)
        if equity == 0:
            return {}

        weights = {}
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in prices:
                value = position.quantity * prices[symbol]
                weights[symbol] = float(value / equity)
        return weights

    def get_total_realized_pnl(self) -> Decimal:
        """计算已实现盈亏总和"""
        return sum(pos.realized_pnl for pos in self.positions.values())

    def get_active_positions(self) -> Dict[str, Position]:
        """获取所有有持仓的标的"""
        return {
            symbol: pos for symbol, pos in self.positions.items()
            if pos.quantity != 0
        }
