# backend/engine/walk_forward.py
"""
Walk-Forward分析：滚动窗口优化验证参数稳定性
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple, Optional
from itertools import product
import pandas as pd

from .models import BacktestResult, WalkForwardResult
from .backtester import Strategy, EventDrivenBacktester

# Re-export WalkForwardResult for convenience
__all__ = ["WalkForwardOptimizer", "WalkForwardResult"]


# Metric name mapping for compatibility
METRIC_ALIASES = {
    "sharpe_ratio": "sharpe",
    "sharpe": "sharpe",
    "calmar_ratio": "calmar",
    "calmar": "calmar",
    "sortino_ratio": "sortino",
    "sortino": "sortino",
    "max_drawdown": "max_drawdown",
    "total_return": "total_return",
    "cagr": "cagr",
}


class WalkForwardOptimizer:
    """Walk-Forward分析：滚动窗口优化验证参数稳定性"""

    def __init__(
        self,
        train_days: int = 252,      # 训练期1年
        test_days: int = 63,        # 测试期1季度
        step_days: int = 63,        # 滚动步长
        anchored: bool = False      # 是否锚定起点
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.anchored = anchored

    def optimize(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_factory: Callable[[Dict], Strategy],
        param_grid: Dict[str, List],
        metric: str = "sharpe_ratio",
        benchmark_data: pd.DataFrame = None
    ) -> WalkForwardResult:
        """
        Run Walk-Forward optimization

        Args:
            data: {symbol: DataFrame} multi-asset OHLCV data
            strategy_factory: function that takes params dict and returns Strategy instance
            param_grid: {"param_name": [values]} search space
            metric: optimization target metric
            benchmark_data: benchmark data

        Returns:
            WalkForwardResult
        """
        splits = self._generate_splits(data)
        all_results = []
        param_history = []

        for i, (train_data, test_data) in enumerate(splits):
            # Find best params on training set
            best_params, best_metric = self._grid_search(
                train_data, strategy_factory, param_grid, metric, benchmark_data
            )
            param_history.append(best_params)

            # Validate on test set
            strategy = strategy_factory(best_params)
            backtester = EventDrivenBacktester(strategy)
            test_result = backtester.run(test_data, benchmark_data)
            test_result.split_index = i
            test_result.optimal_params = best_params
            all_results.append(test_result)

        # Calculate stability score
        stability_score = self._calculate_stability(param_history, param_grid)

        # Combine test period equity curves
        combined_equity = self._combine_equity_curves(all_results)

        return WalkForwardResult(
            combined_equity=combined_equity,
            split_results=all_results,
            param_history=param_history,
            stability_score=stability_score
        )

    def _generate_splits(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[Dict, Dict]]:
        """Generate train/test data splits"""
        all_dates = self._get_common_dates(data)
        n = len(all_dates)
        splits = []

        start = 0
        while start + self.train_days + self.test_days <= n:
            train_start = 0 if self.anchored else start
            train_end = start + self.train_days
            test_end = min(train_end + self.test_days, n)

            train_dates = all_dates[train_start:train_end]
            test_dates = all_dates[train_end:test_end]

            train_data = {sym: df.loc[train_dates[0]:train_dates[-1]] for sym, df in data.items()}
            test_data = {sym: df.loc[test_dates[0]:test_dates[-1]] for sym, df in data.items()}

            splits.append((train_data, test_data))
            start += self.step_days

        return splits

    def _get_common_dates(self, data: Dict[str, pd.DataFrame]) -> List:
        """Get common dates across all symbols"""
        date_sets = [set(df.index) for df in data.values()]
        common = set.intersection(*date_sets) if date_sets else set()
        return sorted(common)

    def _grid_search(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_factory: Callable,
        param_grid: Dict[str, List],
        metric: str,
        benchmark_data: pd.DataFrame
    ) -> Tuple[Dict, float]:
        """Grid search for best parameters"""
        best_params = None
        best_metric = float("-inf")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Map metric name to actual attribute name
        actual_metric = METRIC_ALIASES.get(metric, metric)

        for values in product(*param_values):
            params = dict(zip(param_names, values))
            strategy = strategy_factory(params)
            backtester = EventDrivenBacktester(strategy)
            result = backtester.run(data, benchmark_data)

            metric_value = getattr(result.metrics, actual_metric, 0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params

        return best_params, best_metric

    def _calculate_stability(self, param_history: List[Dict], param_grid: Dict) -> float:
        """Calculate parameter stability score (0-1)"""
        if len(param_history) < 2:
            return 1.0

        stability_scores = []
        for param_name in param_grid.keys():
            values = [p[param_name] for p in param_history]
            unique_ratio = len(set(values)) / len(values)
            stability_scores.append(1 - unique_ratio)

        return sum(stability_scores) / len(stability_scores) if stability_scores else 1.0

    def _combine_equity_curves(self, results: List[BacktestResult]) -> pd.DataFrame:
        """Combine test period equity curves"""
        curves = [r.equity_curve for r in results]
        return pd.concat(curves).sort_index()
