# backend/engine/models.py
"""
回测系统核心数据模型
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    """交易订单"""
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None


@dataclass
class Fill:
    """订单成交记录"""
    order: Order
    fill_price: Decimal
    fill_quantity: Decimal
    commission: Decimal
    slippage: Decimal
    timestamp: datetime


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: Decimal = Decimal("0")
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    entry_date: Optional[date] = None

    def update(self, fill: Fill) -> None:
        """根据成交更新持仓"""
        if fill.order.side == OrderSide.BUY:
            new_quantity = self.quantity + fill.fill_quantity
            if new_quantity != 0:
                self.avg_cost = (
                    (self.quantity * self.avg_cost + fill.fill_quantity * fill.fill_price)
                    / new_quantity
                )
            self.quantity = new_quantity
            if self.entry_date is None:
                self.entry_date = fill.timestamp.date() if fill.timestamp else None
        else:
            self.realized_pnl += fill.fill_quantity * (fill.fill_price - self.avg_cost)
            self.quantity -= fill.fill_quantity
            if self.quantity == 0:
                self.entry_date = None


@dataclass
class Trade:
    """完整交易记录（用于API返回）"""
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    pnl: float = 0.0


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    # 核心指标
    sharpe: float = 0.0
    calmar: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0

    # 收益指标
    total_return: float = 0.0
    cagr: float = 0.0
    ytd_return: float = 0.0
    mtd_return: float = 0.0

    # 风险指标
    volatility: float = 0.0
    alpha: float = 0.0
    beta: float = 1.0

    # 交易统计
    win_rate: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    pl_ratio: float = 0.0
    avg_hold_days: float = 0.0
    consecutive_wins: int = 0

    # 其他
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    drawdown_period: str = "N/A"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容现有API）"""
        return {
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "ytd_return": self.ytd_return,
            "mtd_return": self.mtd_return,
            "volatility": self.volatility,
            "alpha": self.alpha,
            "beta": self.beta,
            "win_rate": self.win_rate,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "pl_ratio": self.pl_ratio,
            "avg_hold_days": self.avg_hold_days,
            "consecutive_wins": self.consecutive_wins,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "drawdown_period": self.drawdown_period,
            # 兼容旧字段名
            "strategy_return": self.cagr,
            "excess_max_drawdown": self.max_drawdown,
        }


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.DataFrame
    trades: List[Fill]
    metrics: PerformanceMetrics
    positions: Dict[str, Position]
    split_index: Optional[int] = None      # Walk-Forward用
    optimal_params: Optional[Dict] = None  # Walk-Forward用


@dataclass
class WalkForwardResult:
    """Walk-Forward分析结果"""
    combined_equity: pd.DataFrame
    split_results: List[BacktestResult]
    param_history: List[Dict]
    stability_score: float


@dataclass
class MonteCarloResult:
    """Monte Carlo分析结果"""
    expected_max_drawdown: float
    var_95: float
    cvar_95: float
    probability_of_loss: Dict[int, float]
    return_confidence_interval: Tuple[float, float]
    simulations: Optional[Any] = None  # numpy array
