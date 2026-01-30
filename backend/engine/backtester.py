# backend/engine/backtester.py
"""
事件驱动回测器：支持多标的、T+1执行、基准对比
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .models import (
    Order, Fill, Position, BacktestResult, PerformanceMetrics
)
from .portfolio import Portfolio
from .execution import ExecutionModel, CompositeExecutionModel
from .metrics import calculate_metrics


class Strategy(ABC):
    """策略抽象基类"""

    @abstractmethod
    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """
        处理K线数据，生成交易订单

        Args:
            timestamp: 当前时间戳
            data: 截止到当前时间的所有数据（避免look-ahead）

        Returns:
            订单列表
        """
        pass

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        """
        处理成交通知

        Args:
            fill: 成交记录
        """
        pass


class EventDrivenBacktester:
    """
    事件驱动回测器

    特点:
    - 支持多标的回测
    - T+1开盘价执行（避免look-ahead bias）
    - 支持基准对比计算Alpha/Beta
    """

    def __init__(
        self,
        strategy: Strategy,
        execution_model: ExecutionModel = None,
        initial_capital: Decimal = Decimal("1000000"),
        benchmark_symbol: str = "000300"
    ):
        self.strategy = strategy
        self.execution_model = execution_model or CompositeExecutionModel()
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol

        # 状态
        self.portfolio: Optional[Portfolio] = None
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.benchmark_curve: List[Tuple[datetime, float]] = []
        self.trades: List[Fill] = []

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame = None
    ) -> BacktestResult:
        """
        运行回测

        Args:
            data: {symbol: DataFrame} 多标的OHLCV数据
            benchmark_data: 基准指数数据

        Returns:
            BacktestResult
        """
        # 初始化
        self.portfolio = Portfolio(
            cash=self.initial_capital,
            initial_capital=self.initial_capital
        )
        self.equity_curve = []
        self.benchmark_curve = []
        self.trades = []

        # 获取所有交易日
        all_dates = self._get_all_dates(data)
        pending_orders: List[Order] = []

        for i, timestamp in enumerate(all_dates):
            # 获取当日各标的数据
            current_bars = self._get_bars_at(data, timestamp)
            next_bars = self._get_bars_at(data, all_dates[i + 1]) if i + 1 < len(all_dates) else {}

            # 执行待处理订单
            new_pending = []
            for order in pending_orders:
                if order.symbol in current_bars:
                    current_bar = current_bars[order.symbol]
                    next_bar = next_bars.get(order.symbol)
                    fill = self.execution_model.execute(order, current_bar, next_bar)
                    if fill:
                        self.portfolio.process_fill(fill)
                        self.strategy.on_fill(fill)
                        self.trades.append(fill)
                    else:
                        new_pending.append(order)
            pending_orders = new_pending

            # 计算当日权益
            prices = {
                symbol: Decimal(str(bar["close"]))
                for symbol, bar in current_bars.items()
            }
            equity = self.portfolio.get_equity(prices)
            self.equity_curve.append((timestamp, float(equity)))

            # 记录基准
            if benchmark_data is not None and timestamp in benchmark_data.index:
                bench_val = benchmark_data.loc[timestamp, "close"]
                self.benchmark_curve.append((timestamp, float(bench_val)))

            # 生成新订单
            combined_data = self._combine_data_up_to(data, timestamp)
            new_orders = self.strategy.on_bar(timestamp, combined_data)
            pending_orders.extend(new_orders)

        return self._create_result(benchmark_data)

    def _get_all_dates(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """获取所有交易日期的并集"""
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        return sorted(all_dates)

    def _get_bars_at(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> Dict[str, pd.Series]:
        """获取指定时间点的所有标的数据"""
        bars = {}
        for symbol, df in data.items():
            if timestamp in df.index:
                bars[symbol] = df.loc[timestamp]
        return bars

    def _combine_data_up_to(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> pd.DataFrame:
        """合并数据到指定时间点"""
        combined = []
        for symbol, df in data.items():
            subset = df.loc[:timestamp].copy()
            subset["symbol"] = symbol
            combined.append(subset)
        return pd.concat(combined) if combined else pd.DataFrame()

    def _create_result(self, benchmark_data: pd.DataFrame = None) -> BacktestResult:
        """创建回测结果"""
        # 构建权益曲线DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        equity_df["returns"] = equity_df["equity"].pct_change()

        # 添加基准
        benchmark_returns = None
        if self.benchmark_curve:
            bench_df = pd.DataFrame(self.benchmark_curve, columns=["timestamp", "benchmark"])
            bench_df.set_index("timestamp", inplace=True)

            # 归一化基准到初始资金
            initial_bench = bench_df["benchmark"].iloc[0]
            bench_df["benchmark"] = bench_df["benchmark"] / initial_bench * float(self.initial_capital)

            equity_df = equity_df.join(bench_df, how="left")
            equity_df["benchmark_returns"] = equity_df["benchmark"].pct_change()
            benchmark_returns = equity_df["benchmark_returns"]

        # 计算绩效指标
        metrics = calculate_metrics(
            equity_df["returns"],
            benchmark_returns=benchmark_returns,
            trades=self.trades
        )

        return BacktestResult(
            equity_curve=equity_df,
            trades=self.trades,
            metrics=metrics,
            positions=dict(self.portfolio.positions)
        )
