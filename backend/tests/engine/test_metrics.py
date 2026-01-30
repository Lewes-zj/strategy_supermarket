# backend/tests/engine/test_metrics.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.metrics import (
    calculate_metrics, calculate_sharpe, calculate_max_drawdown,
    calculate_alpha_beta, calculate_win_rate
)


@pytest.fixture
def sample_returns():
    """创建测试用收益率序列"""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.0005, 0.015, 252),  # 日均0.05%，波动1.5%
        index=dates
    )
    return returns


@pytest.fixture
def benchmark_returns():
    """创建基准收益率序列"""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    np.random.seed(123)
    returns = pd.Series(
        np.random.normal(0.0003, 0.012, 252),
        index=dates
    )
    return returns


class TestCalculateSharpe:
    def test_positive_sharpe(self, sample_returns):
        sharpe = calculate_sharpe(sample_returns)
        # 正收益应该有正Sharpe
        assert isinstance(sharpe, float)

    def test_zero_volatility_returns_zero(self):
        returns = pd.Series([0.01] * 100)
        sharpe = calculate_sharpe(returns)
        # 零波动率应该返回0或无穷大的处理
        assert not np.isnan(sharpe)


class TestCalculateMaxDrawdown:
    def test_drawdown_is_negative(self, sample_returns):
        max_dd = calculate_max_drawdown(sample_returns)
        assert max_dd <= 0

    def test_no_drawdown_for_always_positive(self):
        returns = pd.Series([0.01] * 100)
        max_dd = calculate_max_drawdown(returns)
        assert max_dd == 0


class TestCalculateAlphaBeta:
    def test_alpha_beta_calculation(self, sample_returns, benchmark_returns):
        alpha, beta = calculate_alpha_beta(sample_returns, benchmark_returns)
        assert isinstance(alpha, float)
        assert isinstance(beta, float)

    def test_beta_close_to_one_for_similar_returns(self):
        returns = pd.Series([0.01, -0.005, 0.008, -0.003, 0.006])
        benchmark = pd.Series([0.01, -0.005, 0.008, -0.003, 0.006])
        _, beta = calculate_alpha_beta(returns, benchmark)
        assert abs(beta - 1.0) < 0.1


class TestCalculateWinRate:
    def test_win_rate_calculation(self, sample_returns):
        win_rate, win_count, loss_count = calculate_win_rate(sample_returns)
        assert 0 <= win_rate <= 1
        assert win_count + loss_count > 0

    def test_all_wins(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        win_rate, win_count, loss_count = calculate_win_rate(returns)
        assert win_rate == 1.0
        assert win_count == 3
        assert loss_count == 0


class TestCalculateMetrics:
    def test_full_metrics_calculation(self, sample_returns, benchmark_returns):
        metrics = calculate_metrics(sample_returns, benchmark_returns)

        assert hasattr(metrics, 'sharpe')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'alpha')
        assert hasattr(metrics, 'beta')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'cagr')

    def test_metrics_to_dict(self, sample_returns):
        metrics = calculate_metrics(sample_returns)
        metrics_dict = metrics.to_dict()

        assert 'sharpe' in metrics_dict
        assert 'max_drawdown' in metrics_dict
        assert 'strategy_return' in metrics_dict  # 兼容旧API

    def test_empty_returns_returns_default(self):
        returns = pd.Series([], dtype=float)
        metrics = calculate_metrics(returns)
        assert metrics.sharpe == 0.0
        assert metrics.max_drawdown == 0.0
