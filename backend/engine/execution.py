# backend/engine/execution.py
"""
订单执行模型：支持市价单、限价单、止损单
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime
from typing import Optional
import pandas as pd

from .models import Order, OrderType, OrderSide, Fill


class ExecutionModel(ABC):
    """执行模型抽象基类"""

    @abstractmethod
    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        """
        执行订单

        Args:
            order: 待执行订单
            current_bar: 当前K线
            next_bar: 下一根K线（T+1执行用）

        Returns:
            成交记录，未成交返回None
        """
        pass


class MarketExecutionModel(ExecutionModel):
    """
    市价单执行模型

    T日信号 → T+1开盘执行（避免look-ahead bias）
    """

    def __init__(self, slippage_bps: float = 10, commission_rate: float = 0.0003):
        """
        Args:
            slippage_bps: 滑点（基点），默认10bps
            commission_rate: 佣金费率，默认万三
        """
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type != OrderType.MARKET:
            return None
        if next_bar is None:
            return None  # 无下一根K线，无法T+1执行

        # 使用下一根K线的开盘价
        base_price = Decimal(str(next_bar["open"]))

        # 应用滑点
        slippage_mult = Decimal(str(1 + self.slippage_bps / 10000))
        if order.side == OrderSide.BUY:
            fill_price = base_price * slippage_mult
        else:
            fill_price = base_price / slippage_mult

        # 计算佣金（最低5元）
        trade_value = fill_price * order.quantity
        commission = trade_value * Decimal(str(self.commission_rate))
        commission = max(commission, Decimal("5"))

        # 计算滑点金额
        slippage_amount = abs(fill_price - base_price) * order.quantity

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=slippage_amount,
            timestamp=next_bar.name if hasattr(next_bar, 'name') and isinstance(next_bar.name, datetime) else datetime.now()
        )


class LimitExecutionModel(ExecutionModel):
    """
    限价单执行模型

    价格触及限价时成交
    """

    def __init__(self, commission_rate: float = 0.0003):
        self.commission_rate = commission_rate

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            return None

        bar = next_bar if next_bar is not None else current_bar
        low = Decimal(str(bar["low"]))
        high = Decimal(str(bar["high"]))

        # 检查价格是否触及
        if order.side == OrderSide.BUY:
            if low <= order.limit_price:
                fill_price = order.limit_price
            else:
                return None
        else:
            if high >= order.limit_price:
                fill_price = order.limit_price
            else:
                return None

        # 计算佣金
        trade_value = fill_price * order.quantity
        commission = max(trade_value * Decimal(str(self.commission_rate)), Decimal("5"))

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=Decimal("0"),
            timestamp=bar.name if hasattr(bar, 'name') and isinstance(bar.name, datetime) else datetime.now()
        )


class StopExecutionModel(ExecutionModel):
    """
    止损单执行模型

    价格突破止损价时以市价成交
    """

    def __init__(self, slippage_bps: float = 20, commission_rate: float = 0.0003):
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type != OrderType.STOP or order.stop_price is None:
            return None

        bar = next_bar if next_bar is not None else current_bar
        low = Decimal(str(bar["low"]))
        high = Decimal(str(bar["high"]))
        open_price = Decimal(str(bar["open"]))

        # 检查是否触发止损
        triggered = False
        if order.side == OrderSide.SELL:  # 止损卖出
            if low <= order.stop_price:
                triggered = True
        else:  # 止损买入（做空回补）
            if high >= order.stop_price:
                triggered = True

        if not triggered:
            return None

        # 触发后以开盘价成交，带滑点
        slippage_mult = Decimal(str(1 + self.slippage_bps / 10000))
        if order.side == OrderSide.BUY:
            fill_price = open_price * slippage_mult
        else:
            fill_price = open_price / slippage_mult

        trade_value = fill_price * order.quantity
        commission = max(trade_value * Decimal(str(self.commission_rate)), Decimal("5"))
        slippage_amount = abs(fill_price - open_price) * order.quantity

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=slippage_amount,
            timestamp=bar.name if hasattr(bar, 'name') and isinstance(bar.name, datetime) else datetime.now()
        )


class CompositeExecutionModel(ExecutionModel):
    """
    组合执行模型

    根据订单类型自动选择对应的执行器
    """

    def __init__(
        self,
        slippage_bps: float = 10,
        stop_slippage_bps: float = 20,
        commission_rate: float = 0.0003
    ):
        self.market_model = MarketExecutionModel(slippage_bps, commission_rate)
        self.limit_model = LimitExecutionModel(commission_rate)
        self.stop_model = StopExecutionModel(stop_slippage_bps, commission_rate)

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type == OrderType.MARKET:
            return self.market_model.execute(order, current_bar, next_bar)
        elif order.order_type == OrderType.LIMIT:
            return self.limit_model.execute(order, current_bar, next_bar)
        elif order.order_type == OrderType.STOP:
            return self.stop_model.execute(order, current_bar, next_bar)
        return None
