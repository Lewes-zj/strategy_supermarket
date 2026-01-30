# backend/tests/engine/test_walk_forward.py
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import List, Dict

from engine.models import Order, OrderSide, OrderType, Fill, BacktestResult
from engine.backtester import Strategy
from engine.walk_forward import WalkForwardOptimizer, WalkForwardResult


class ParameterizedTestStrategy(Strategy):
    """参数化测试策略"""

    def __init__(self, lookback: int = 10, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold
        self.position = 0

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        orders = []
        if len(data) < self.lookback:
            return orders

        # 简单动量策略
        recent = data.tail(self.lookback)
        if "close" in recent.columns:
            returns = recent["close"].pct_change().mean()
            if returns > self.threshold and self.position == 0:
                orders.append(Order(
                    symbol="000001",
                    side=OrderSide.BUY,
                    quantity=Decimal("100"),
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                ))
            elif returns < -self.threshold and self.position > 0:
                orders.append(Order(
                    symbol="000001",
                    side=OrderSide.SELL,
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
    """创建测试用数据（2年）"""
    dates = pd.date_range(start="2022-01-01", periods=504, freq="B")  # 2年交易日
    np.random.seed(42)

    # 生成有趋势的价格数据
    trend = np.linspace(10, 15, 504)
    noise = np.random.normal(0, 0.3, 504)
    prices = trend + noise

    data_000001 = pd.DataFrame({
        "open": prices + np.random.normal(0, 0.1, 504),
        "high": prices + abs(np.random.normal(0.2, 0.1, 504)),
        "low": prices - abs(np.random.normal(0.2, 0.1, 504)),
        "close": prices,
        "volume": np.random.randint(1000000, 2000000, 504)
    }, index=dates)

    return {"000001": data_000001}


@pytest.fixture
def param_grid():
    """参数搜索空间"""
    return {
        "lookback": [5, 10, 20],
        "threshold": [0.01, 0.02, 0.03]
    }


def strategy_factory(params: Dict) -> Strategy:
    """策略工厂函数"""
    return ParameterizedTestStrategy(**params)


class TestWalkForwardOptimizer:
    def test_generate_splits(self, sample_data):
        optimizer = WalkForwardOptimizer(
            train_days=126,  # 半年
            test_days=63,    # 一季度
            step_days=63     # 滚动一季度
        )

        splits = optimizer._generate_splits(sample_data)

        # 2年数据应该生成多个分割
        assert len(splits) >= 2

        # 每个分割应该有训练集和测试集
        for train_data, test_data in splits:
            assert len(train_data) > 0
            assert len(test_data) > 0

    def test_optimize_returns_result(self, sample_data, param_grid):
        optimizer = WalkForwardOptimizer(
            train_days=126,
            test_days=63,
            step_days=63
        )

        result = optimizer.optimize(
            data=sample_data,
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            metric="sharpe_ratio"
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.param_history) > 0
        assert result.combined_equity is not None
        assert 0 <= result.stability_score <= 1

    def test_stability_score_calculation(self, sample_data, param_grid):
        optimizer = WalkForwardOptimizer(
            train_days=126,
            test_days=63,
            step_days=63
        )

        result = optimizer.optimize(
            data=sample_data,
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            metric="sharpe_ratio"
        )

        # 稳定性评分应在0-1之间
        assert 0 <= result.stability_score <= 1

    def test_anchored_mode(self, sample_data, param_grid):
        optimizer = WalkForwardOptimizer(
            train_days=126,
            test_days=63,
            step_days=63,
            anchored=True  # 锚定模式
        )

        splits = optimizer._generate_splits(sample_data)

        # 锚定模式下，所有训练集应从头开始
        for i, (train_data, test_data) in enumerate(splits):
            # 第一个标的的数据
            first_symbol = list(train_data.keys())[0]
            train_df = train_data[first_symbol]
            # 训练数据应该从同一起点开始
            if i > 0:
                first_symbol_prev = list(splits[0][0].keys())[0]
                assert train_df.index[0] == splits[0][0][first_symbol_prev].index[0]

    def test_split_results_contains_backtest_results(self, sample_data, param_grid):
        optimizer = WalkForwardOptimizer(
            train_days=126,
            test_days=63,
            step_days=63
        )

        result = optimizer.optimize(
            data=sample_data,
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            metric="sharpe_ratio"
        )

        # 每个分割应该有对应的回测结果
        assert len(result.split_results) == len(result.param_history)
        for split_result in result.split_results:
            assert isinstance(split_result, BacktestResult)
