# services/backtest_service.py
"""回测服务层：统一调度回测、Walk-Forward和Monte Carlo分析"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Callable
import hashlib
import json
import pandas as pd
import numpy as np

from engine.models import BacktestResult, WalkForwardResult, MonteCarloResult
from engine.backtester import EventDrivenBacktester, Strategy
from engine.walk_forward import WalkForwardOptimizer
from engine.monte_carlo import MonteCarloAnalyzer


class BacktestCache:
    """Simple in-memory cache for backtest results"""

    def __init__(self, ttl_hours: int = 24):
        self.ttl_hours = ttl_hours
        self._cache: Dict[str, BacktestResult] = {}

    def get(self, key: str) -> Optional[BacktestResult]:
        return self._cache.get(key)

    def set(self, key: str, value: BacktestResult) -> None:
        self._cache[key] = value

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)


class BacktestService:
    """回测服务层：统一调度"""

    def __init__(self, data_service=None):
        self.data_service = data_service
        self.cache = BacktestCache(ttl_hours=24)
        self._strategies: Dict[str, type] = {}  # strategy_id -> Strategy class

    def register_strategy(self, strategy_id: str, strategy_class: type) -> None:
        """注册策略"""
        self._strategies[strategy_id] = strategy_class

    def get_strategy_factory(self, strategy_id: str) -> Optional[type]:
        """获取策略工厂（类）"""
        return self._strategies.get(strategy_id)

    def run_backtest(
        self,
        strategy_id: str,
        symbols: List[str] = None,
        start_date: date = None,
        end_date: date = None,
        **strategy_params
    ) -> BacktestResult:
        """
        标准回测

        Args:
            strategy_id: 策略ID
            symbols: 标的列表
            start_date: 开始日期
            end_date: 结束日期
            **strategy_params: 策略参数

        Returns:
            BacktestResult
        """
        # 默认10年数据
        if start_date is None:
            start_date = date.today() - timedelta(days=365 * 10)
        if end_date is None:
            end_date = date.today()
        if symbols is None:
            symbols = ["000001"]

        # 验证参数
        if start_date > end_date:
            raise ValueError("Invalid date range: start_date cannot be after end_date")
        if not symbols:
            raise ValueError("Symbols list cannot be empty")

        # 检查缓存 - include strategy_params in cache key
        params_json = json.dumps(strategy_params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_json.encode()).hexdigest()
        cache_key = f"{strategy_id}:{','.join(sorted(symbols))}:{start_date}:{end_date}:{params_hash}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # 加载数据
        data = self._get_data(symbols, start_date, end_date)
        benchmark = self._get_benchmark(start_date, end_date)

        # 执行回测
        strategy = self._create_strategy(strategy_id, **strategy_params)
        backtester = EventDrivenBacktester(strategy, benchmark_symbol="000300")
        result = backtester.run(data, benchmark)

        # 缓存
        self.cache.set(cache_key, result)
        return result

    def run_walk_forward(
        self,
        strategy_id: str,
        param_grid: Dict[str, List],
        train_days: int = 252,
        test_days: int = 63,
        symbols: List[str] = None,
        start_date: date = None,
        end_date: date = None
    ) -> WalkForwardResult:
        """
        Walk-Forward分析

        Args:
            strategy_id: 策略ID
            param_grid: 参数搜索空间
            train_days: 训练期天数
            test_days: 测试期天数

        Returns:
            WalkForwardResult
        """
        if symbols is None:
            symbols = ["000001"]
        if start_date is None:
            start_date = date.today() - timedelta(days=365 * 10)
        if end_date is None:
            end_date = date.today()

        # 加载数据
        data = self._get_data(symbols, start_date, end_date)
        benchmark = self._get_benchmark(start_date, end_date)

        # 创建策略工厂
        def strategy_factory(params: Dict) -> Strategy:
            return self._create_strategy(strategy_id, **params)

        optimizer = WalkForwardOptimizer(train_days, test_days)
        return optimizer.optimize(data, strategy_factory, param_grid, benchmark_data=benchmark)

    def run_monte_carlo(
        self,
        strategy_id: str,
        n_simulations: int = 1000,
        symbols: List[str] = None,
        start_date: date = None,
        end_date: date = None
    ) -> MonteCarloResult:
        """
        Monte Carlo分析

        Args:
            strategy_id: 策略ID
            n_simulations: 模拟次数

        Returns:
            MonteCarloResult
        """
        result = self.run_backtest(strategy_id, symbols, start_date, end_date)
        analyzer = MonteCarloAnalyzer(n_simulations)
        return analyzer.analyze(result.equity_curve["returns"])

    def _get_data(self, symbols: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """获取历史数据"""
        if self.data_service:
            return self.data_service.get_data_for_backtest(symbols, start_date, end_date)

        # Fallback: generate mock data for testing
        return self._generate_mock_data(symbols, start_date, end_date)

    def _get_benchmark(self, start_date: date, end_date: date) -> pd.DataFrame:
        """获取基准数据"""
        if self.data_service:
            return self.data_service.get_benchmark_data("000300", start_date, end_date)

        # Fallback: generate mock benchmark
        return self._generate_mock_benchmark(start_date, end_date)

    def _create_strategy(self, strategy_id: str, **params) -> Strategy:
        """创建策略实例"""
        if strategy_id not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        return self._strategies[strategy_id](**params)

    def _generate_mock_data(self, symbols: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """生成模拟数据（测试用）"""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        data = {}
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            prices = 100 + np.cumsum(np.random.normal(0.1, 2, len(dates)))
            data[symbol] = pd.DataFrame({
                "open": prices + np.random.normal(0, 0.5, len(dates)),
                "high": prices + abs(np.random.normal(0.5, 0.3, len(dates))),
                "low": prices - abs(np.random.normal(0.5, 0.3, len(dates))),
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
        return data

    def _generate_mock_benchmark(self, start_date: date, end_date: date) -> pd.DataFrame:
        """生成模拟基准（测试用）"""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        np.random.seed(42)
        prices = 3000 + np.cumsum(np.random.normal(0.5, 10, len(dates)))
        return pd.DataFrame({
            "open": prices + np.random.normal(0, 2, len(dates)),
            "high": prices + abs(np.random.normal(2, 1, len(dates))),
            "low": prices - abs(np.random.normal(2, 1, len(dates))),
            "close": prices,
            "volume": np.random.randint(10000000, 50000000, len(dates))
        }, index=dates)
