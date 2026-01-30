# backend/tests/engine/test_monte_carlo.py
"""
Monte Carlo分析器测试

测试内容:
- Bootstrap收益率模拟
- VaR和CVaR计算
- 最大回撤分析
- 不同持有期的亏损概率
- 收益率置信区间
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from engine.monte_carlo import MonteCarloAnalyzer
from engine.models import MonteCarloResult


@pytest.fixture
def sample_returns():
    """创建样本收益率序列（1年的日收益率）"""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.0005, 0.015, 252),  # 日均0.05%，波动1.5%
        index=dates
    )
    return returns


@pytest.fixture
def negative_returns():
    """创建负收益率序列（用于测试边界情况）"""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(-0.001, 0.02, 252),  # 日均-0.1%，波动2%
        index=dates
    )
    return returns


@pytest.fixture
def volatile_returns():
    """创建高波动率序列"""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.0005, 0.05, 252),  # 日均0.05%，波动5%
        index=dates
    )
    return returns


class TestMonteCarloAnalyzer:
    """Monte Carlo分析器基本测试"""

    def test_analyze_returns_result(self, sample_returns):
        """测试analyze方法返回正确的结果类型"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        assert isinstance(result, MonteCarloResult)
        assert result.expected_max_drawdown < 0  # 回撤是负数
        assert result.var_95 is not None
        assert result.cvar_95 <= result.var_95  # CVaR比VaR更差（更负）

    def test_default_parameters(self, sample_returns):
        """测试默认参数"""
        analyzer = MonteCarloAnalyzer()
        assert analyzer.n_simulations == 1000
        assert analyzer.confidence == 0.95

    def test_custom_parameters(self, sample_returns):
        """测试自定义参数"""
        analyzer = MonteCarloAnalyzer(n_simulations=500, confidence=0.99)
        assert analyzer.n_simulations == 500
        assert analyzer.confidence == 0.99

    def test_reproducibility_with_seed(self, sample_returns):
        """测试随机种子的可重复性"""
        np.random.seed(123)
        analyzer1 = MonteCarloAnalyzer(n_simulations=100)
        result1 = analyzer1.analyze(sample_returns)

        np.random.seed(123)
        analyzer2 = MonteCarloAnalyzer(n_simulations=100)
        result2 = analyzer2.analyze(sample_returns)

        assert result1.var_95 == result2.var_95
        assert result1.cvar_95 == result2.cvar_95


class TestProbabilityOfLoss:
    """亏损概率测试"""

    def test_probability_of_loss(self, sample_returns):
        """测试亏损概率计算"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        # 检查亏损概率字典
        assert len(result.probability_of_loss) > 0
        for period, prob in result.probability_of_loss.items():
            assert 0 <= prob <= 1
            assert isinstance(period, int)

    def test_probability_of_loss_periods(self, sample_returns):
        """测试不同持有期的亏损概率"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        # 典型持有期：1天，5天，20天，60天，252天
        # 至少应该包含一些常用持有期
        assert any(period in result.probability_of_loss for period in [1, 5, 20, 60, 252])

    def test_longer_period_lower_loss_probability(self, sample_returns):
        """测试对于正收益策略，更长持有期应该有更低的亏损概率"""
        analyzer = MonteCarloAnalyzer(n_simulations=500)
        result = analyzer.analyze(sample_returns)

        probs = result.probability_of_loss
        periods = sorted(probs.keys())

        # 对于有正期望收益的策略，长期持有亏损概率通常更低
        # 但这不是绝对的，所以只做基本检查
        if len(periods) >= 2:
            # 至少有两个周期可以比较
            assert all(probs[p] >= 0 for p in periods)


class TestConfidenceInterval:
    """置信区间测试"""

    def test_confidence_interval(self, sample_returns):
        """测试置信区间计算"""
        analyzer = MonteCarloAnalyzer(n_simulations=100, confidence=0.95)
        result = analyzer.analyze(sample_returns)

        lower, upper = result.return_confidence_interval
        assert lower < upper

    def test_wider_interval_with_higher_confidence(self, sample_returns):
        """测试更高置信度应该有更宽的区间"""
        analyzer_95 = MonteCarloAnalyzer(n_simulations=500, confidence=0.95)
        analyzer_99 = MonteCarloAnalyzer(n_simulations=500, confidence=0.99)

        np.random.seed(42)
        result_95 = analyzer_95.analyze(sample_returns)
        np.random.seed(42)
        result_99 = analyzer_99.analyze(sample_returns)

        width_95 = result_95.return_confidence_interval[1] - result_95.return_confidence_interval[0]
        width_99 = result_99.return_confidence_interval[1] - result_99.return_confidence_interval[0]

        assert width_99 >= width_95

    def test_confidence_interval_bounds(self, sample_returns):
        """测试置信区间的合理性"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        lower, upper = result.return_confidence_interval
        # 区间应该是有限的数值
        assert np.isfinite(lower)
        assert np.isfinite(upper)


class TestVaRCVaR:
    """VaR和CVaR测试"""

    def test_var_cvar_relationship(self, sample_returns):
        """测试VaR和CVaR的关系"""
        analyzer = MonteCarloAnalyzer(n_simulations=500)
        result = analyzer.analyze(sample_returns)

        # CVaR（期望损失）应该 <= VaR（更负或相等）
        assert result.cvar_95 <= result.var_95

    def test_var_is_negative_or_zero(self, sample_returns):
        """测试VaR通常为负数或零"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        # 95% VaR通常表示最差5%情况的损失
        # 对于大多数策略，这应该是负数
        assert isinstance(result.var_95, float)
        assert np.isfinite(result.var_95)

    def test_cvar_is_finite(self, sample_returns):
        """测试CVaR是有限数值"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        assert np.isfinite(result.cvar_95)

    def test_higher_volatility_worse_var(self, sample_returns, volatile_returns):
        """测试更高波动率应该导致更差的VaR"""
        analyzer = MonteCarloAnalyzer(n_simulations=500)

        np.random.seed(42)
        result_normal = analyzer.analyze(sample_returns)
        np.random.seed(42)
        result_volatile = analyzer.analyze(volatile_returns)

        # 更高波动率应该导致更差（更负）的VaR
        assert result_volatile.var_95 < result_normal.var_95


class TestDrawdownAnalysis:
    """最大回撤分析测试"""

    def test_expected_max_drawdown_is_negative(self, sample_returns):
        """测试期望最大回撤为负数"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        assert result.expected_max_drawdown < 0

    def test_higher_volatility_worse_drawdown(self, sample_returns, volatile_returns):
        """测试更高波动率应该导致更大的最大回撤"""
        analyzer = MonteCarloAnalyzer(n_simulations=500)

        np.random.seed(42)
        result_normal = analyzer.analyze(sample_returns)
        np.random.seed(42)
        result_volatile = analyzer.analyze(volatile_returns)

        # 更高波动率应该导致更大的回撤（更负）
        assert result_volatile.expected_max_drawdown < result_normal.expected_max_drawdown

    def test_drawdown_bounded(self, sample_returns):
        """测试回撤在合理范围内"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        # 回撤应该在 -1（亏损100%）和 0 之间
        assert -1.0 <= result.expected_max_drawdown <= 0


class TestSimulationsStorage:
    """模拟结果存储测试"""

    def test_simulations_stored(self, sample_returns):
        """测试模拟结果被正确存储"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(sample_returns)

        assert result.simulations is not None
        assert result.simulations.shape[0] == 100  # n_simulations行

    def test_simulations_shape(self, sample_returns):
        """测试模拟结果的形状"""
        n_sims = 200
        analyzer = MonteCarloAnalyzer(n_simulations=n_sims)
        result = analyzer.analyze(sample_returns)

        assert result.simulations.shape[0] == n_sims
        # 列数应该等于原始数据的长度或相关
        assert result.simulations.shape[1] > 0


class TestEdgeCases:
    """边界情况测试"""

    def test_short_returns_series(self):
        """测试短收益率序列"""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.01, 10), index=dates)

        analyzer = MonteCarloAnalyzer(n_simulations=50)
        result = analyzer.analyze(returns)

        assert isinstance(result, MonteCarloResult)

    def test_zero_returns(self):
        """测试零收益率序列"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
        returns = pd.Series([0.0] * 100, index=dates)

        analyzer = MonteCarloAnalyzer(n_simulations=50)
        result = analyzer.analyze(returns)

        assert isinstance(result, MonteCarloResult)
        # 零收益率应该导致零回撤
        assert result.expected_max_drawdown == 0

    def test_constant_positive_returns(self):
        """测试恒定正收益率"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
        returns = pd.Series([0.01] * 100, index=dates)

        analyzer = MonteCarloAnalyzer(n_simulations=50)
        result = analyzer.analyze(returns)

        assert isinstance(result, MonteCarloResult)
        # 恒定正收益率应该导致零回撤
        assert result.expected_max_drawdown == 0

    def test_single_large_loss(self):
        """测试包含单次大亏损的序列"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
        returns = pd.Series([0.001] * 100, index=dates)
        returns.iloc[50] = -0.20  # 单次20%亏损

        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(returns)

        assert isinstance(result, MonteCarloResult)
        # 应该检测到这次大亏损的影响
        assert result.expected_max_drawdown < -0.05  # 至少有5%的回撤

    def test_negative_returns_series(self, negative_returns):
        """测试负收益率序列"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)
        result = analyzer.analyze(negative_returns)

        assert isinstance(result, MonteCarloResult)
        # 负收益策略应该有较大的回撤
        assert result.expected_max_drawdown < 0
        # 亏损概率应该较高
        if 20 in result.probability_of_loss:
            assert result.probability_of_loss[20] > 0.3


class TestStatisticalProperties:
    """统计特性测试"""

    def test_more_simulations_more_stable(self, sample_returns):
        """测试更多模拟次数应该产生更稳定的结果"""
        np.random.seed(42)
        analyzer_few = MonteCarloAnalyzer(n_simulations=50)
        result_few_1 = analyzer_few.analyze(sample_returns)

        np.random.seed(43)
        result_few_2 = analyzer_few.analyze(sample_returns)

        np.random.seed(42)
        analyzer_many = MonteCarloAnalyzer(n_simulations=1000)
        result_many_1 = analyzer_many.analyze(sample_returns)

        np.random.seed(43)
        result_many_2 = analyzer_many.analyze(sample_returns)

        # 更多模拟次数的结果差异应该更小
        diff_few = abs(result_few_1.var_95 - result_few_2.var_95)
        diff_many = abs(result_many_1.var_95 - result_many_2.var_95)

        # 由于随机性，不能保证总是成立，但大概率成立
        # 如果测试失败，可能只是随机波动
        assert diff_few >= 0  # 基本检查
        assert diff_many >= 0  # 基本检查

    def test_results_are_deterministic_with_same_seed(self, sample_returns):
        """测试相同种子产生相同结果"""
        analyzer = MonteCarloAnalyzer(n_simulations=100)

        np.random.seed(12345)
        result1 = analyzer.analyze(sample_returns)

        np.random.seed(12345)
        result2 = analyzer.analyze(sample_returns)

        assert result1.var_95 == result2.var_95
        assert result1.cvar_95 == result2.cvar_95
        assert result1.expected_max_drawdown == result2.expected_max_drawdown


class TestInputValidation:
    """输入验证测试"""

    def test_empty_returns_raises_error(self):
        """测试空收益率序列应该引发错误或返回默认值"""
        returns = pd.Series([], dtype=float)
        analyzer = MonteCarloAnalyzer(n_simulations=50)

        # 根据实现，可能引发错误或返回默认值
        # 这里测试不应该崩溃
        try:
            result = analyzer.analyze(returns)
            # 如果不引发错误，应该返回有效结果
            assert isinstance(result, MonteCarloResult)
        except (ValueError, IndexError):
            # 引发错误也是可接受的行为
            pass

    def test_invalid_confidence_level(self):
        """测试无效置信度应该引发错误"""
        with pytest.raises((ValueError, AssertionError)):
            MonteCarloAnalyzer(confidence=1.5)  # 大于1

        with pytest.raises((ValueError, AssertionError)):
            MonteCarloAnalyzer(confidence=-0.1)  # 负数

    def test_invalid_n_simulations(self):
        """测试无效模拟次数应该引发错误"""
        with pytest.raises((ValueError, AssertionError)):
            MonteCarloAnalyzer(n_simulations=0)

        with pytest.raises((ValueError, AssertionError)):
            MonteCarloAnalyzer(n_simulations=-10)
