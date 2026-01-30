# backend/engine/monte_carlo.py
"""
Monte Carlo模拟分析：评估策略风险和收益置信区间
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .models import MonteCarloResult


class MonteCarloAnalyzer:
    """Monte Carlo模拟分析：评估策略风险和收益置信区间"""

    def __init__(self, n_simulations: int = 1000, confidence: float = 0.95):
        """
        初始化Monte Carlo分析器

        Args:
            n_simulations: 模拟次数
            confidence: 置信度 (0 < confidence < 1)
        """
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if not (0 < confidence < 1):
            raise ValueError("confidence must be between 0 and 1 (exclusive)")

        self.n_simulations = n_simulations
        self.confidence = confidence

    def analyze(self, returns: pd.Series) -> MonteCarloResult:
        """
        完整Monte Carlo分析

        Args:
            returns: 历史收益率序列

        Returns:
            MonteCarloResult
        """
        # Validate input
        returns = returns.dropna()
        if len(returns) == 0:
            raise ValueError("Returns series cannot be empty")

        # Bootstrap模拟
        simulations = self._bootstrap_returns(returns)

        # 分析最大回撤分布
        drawdown_stats = self._analyze_drawdowns(simulations)

        # 计算VaR和CVaR
        var_95, cvar_95 = self._calculate_var(simulations)

        # 各持有期亏损概率
        prob_loss = self._probability_of_loss(returns, [21, 63, 126, 252])

        # 收益置信区间
        confidence_interval = self._return_confidence_interval(simulations)

        return MonteCarloResult(
            expected_max_drawdown=drawdown_stats["expected"],
            var_95=var_95,
            cvar_95=cvar_95,
            probability_of_loss=prob_loss,
            return_confidence_interval=confidence_interval,
            simulations=simulations
        )

    def _bootstrap_returns(self, returns: pd.Series, n_periods: int = None) -> np.ndarray:
        """Bootstrap重采样模拟"""
        if n_periods is None:
            n_periods = len(returns)

        simulations = np.zeros((self.n_simulations, n_periods))
        returns_arr = returns.dropna().values

        for i in range(self.n_simulations):
            simulations[i] = np.random.choice(returns_arr, size=n_periods, replace=True)

        return simulations

    def _analyze_drawdowns(self, simulations: np.ndarray) -> Dict[str, float]:
        """分析回撤分布"""
        max_drawdowns = []

        for sim_returns in simulations:
            equity = (1 + sim_returns).cumprod()
            rolling_max = np.maximum.accumulate(equity)
            drawdowns = (equity - rolling_max) / rolling_max
            max_drawdowns.append(drawdowns.min())

        max_drawdowns = np.array(max_drawdowns)

        return {
            "expected": float(np.mean(max_drawdowns)),
            "median": float(np.median(max_drawdowns)),
            "worst_95pct": float(np.percentile(max_drawdowns, 5)),
            "worst_case": float(max_drawdowns.min())
        }

    def _calculate_var(self, simulations: np.ndarray) -> Tuple[float, float]:
        """计算VaR和CVaR"""
        total_returns = (1 + simulations).prod(axis=1) - 1
        var_95 = float(np.percentile(total_returns, 5))
        tail_returns = total_returns[total_returns <= var_95]
        cvar_95 = float(tail_returns.mean()) if len(tail_returns) > 0 else var_95
        return var_95, cvar_95

    def _probability_of_loss(
        self,
        returns: pd.Series,
        holding_periods: List[int]
    ) -> Dict[int, float]:
        """计算各持有期亏损概率"""
        results = {}

        for period in holding_periods:
            if period > len(returns):
                continue
            simulations = self._bootstrap_returns(returns, period)
            total_returns = (1 + simulations).prod(axis=1) - 1
            prob_loss = float((total_returns < 0).mean())
            results[period] = prob_loss

        return results

    def _return_confidence_interval(self, simulations: np.ndarray) -> Tuple[float, float]:
        """计算收益置信区间"""
        total_returns = (1 + simulations).prod(axis=1) - 1
        alpha = (1 - self.confidence) * 100  # e.g., 5 for 95% confidence
        lower = alpha / 2                     # e.g., 2.5
        upper = 100 - alpha / 2               # e.g., 97.5
        return (
            float(np.percentile(total_returns, lower)),
            float(np.percentile(total_returns, upper))
        )
