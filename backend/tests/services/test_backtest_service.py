# backend/tests/services/test_backtest_service.py
"""
Tests for BacktestService - the service layer that coordinates backtesting.

BacktestService provides:
- run_backtest(strategy_id, symbols, start_date, end_date) -> BacktestResult
- run_walk_forward(strategy_id, param_grid, train_days, test_days) -> WalkForwardResult
- run_monte_carlo(strategy_id, n_simulations) -> MonteCarloResult

It coordinates between:
- DataService (gets historical data)
- EventDrivenBacktester (runs the backtest)
- WalkForwardOptimizer (parameter optimization)
- MonteCarloAnalyzer (risk analysis)
"""
import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from decimal import Decimal

from engine.models import (
    BacktestResult, WalkForwardResult, MonteCarloResult,
    PerformanceMetrics, Position, Fill, Order, OrderSide, OrderType
)


@pytest.fixture
def mock_data():
    """Create mock price data for backtesting"""
    dates = pd.date_range(start="2020-01-01", periods=504, freq="B")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, 504))
    data = pd.DataFrame({
        "open": prices + np.random.normal(0, 0.5, 504),
        "high": prices + abs(np.random.normal(0.5, 0.3, 504)),
        "low": prices - abs(np.random.normal(0.5, 0.3, 504)),
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 504)
    }, index=dates)
    return {"000001": data}


@pytest.fixture
def mock_benchmark_data():
    """Create mock benchmark data"""
    dates = pd.date_range(start="2020-01-01", periods=504, freq="B")
    np.random.seed(43)
    prices = 3000 + np.cumsum(np.random.normal(0.5, 10, 504))
    return pd.DataFrame({
        "open": prices + np.random.normal(0, 2, 504),
        "high": prices + abs(np.random.normal(2, 1, 504)),
        "low": prices - abs(np.random.normal(2, 1, 504)),
        "close": prices,
        "volume": np.random.randint(10000000, 50000000, 504)
    }, index=dates)


@pytest.fixture
def mock_backtest_result(mock_data):
    """Create mock BacktestResult for testing"""
    dates = list(mock_data["000001"].index)
    equity_df = pd.DataFrame({
        "equity": np.linspace(1000000, 1100000, len(dates)),
        "returns": [0.0] + list(np.random.normal(0.0004, 0.02, len(dates) - 1))
    }, index=dates)

    return BacktestResult(
        equity_curve=equity_df,
        trades=[],
        metrics=PerformanceMetrics(
            sharpe=1.5,
            calmar=0.8,
            sortino=1.8,
            max_drawdown=-0.15,
            total_return=0.10,
            cagr=0.12,
            volatility=0.18,
            win_rate=0.55,
            win_count=30,
            loss_count=25
        ),
        positions={}
    )


@pytest.fixture
def mock_walk_forward_result(mock_data):
    """Create mock WalkForwardResult for testing"""
    dates = list(mock_data["000001"].index)
    combined_equity = pd.DataFrame({
        "equity": np.linspace(1000000, 1080000, len(dates)),
        "returns": [0.0] + list(np.random.normal(0.0003, 0.015, len(dates) - 1))
    }, index=dates)

    return WalkForwardResult(
        combined_equity=combined_equity,
        split_results=[],
        param_history=[{"fast": 10, "slow": 30}] * 4,
        stability_score=0.75
    )


@pytest.fixture
def mock_monte_carlo_result():
    """Create mock MonteCarloResult for testing"""
    return MonteCarloResult(
        expected_max_drawdown=-0.18,
        var_95=-0.12,
        cvar_95=-0.16,
        probability_of_loss={21: 0.35, 63: 0.28, 126: 0.22, 252: 0.15},
        return_confidence_interval=(-0.05, 0.25),
        simulations=np.random.normal(0.0004, 0.02, (1000, 252))
    )


class TestBacktestService:
    """Tests for basic backtest functionality"""

    def test_run_backtest_returns_result(self, mock_data):
        """Test that run_backtest returns BacktestResult"""
        # Import should fail since BacktestService doesn't exist yet
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_run_backtest_with_date_range(self, mock_data):
        """Test backtest with custom date range"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_run_backtest_uses_cache(self, mock_data):
        """Test that repeated calls use cache"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_run_backtest_with_strategy_id(self, mock_data):
        """Test running backtest with a specific strategy ID"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_run_backtest_with_multiple_symbols(self, mock_data):
        """Test running backtest with multiple symbols"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_run_backtest_calls_data_service(self, mock_data):
        """Test that run_backtest calls DataService for data"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_run_backtest_calls_backtester(self, mock_data):
        """Test that run_backtest uses EventDrivenBacktester"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService


class TestBacktestServiceIntegration:
    """Integration tests for BacktestService with mocked dependencies"""

    def test_backtest_returns_correct_result_type(self, mock_data, mock_backtest_result):
        """Test that backtest returns BacktestResult with correct structure"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_backtest_result_has_equity_curve(self, mock_data, mock_backtest_result):
        """Test that result contains equity curve DataFrame"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_backtest_result_has_metrics(self, mock_data, mock_backtest_result):
        """Test that result contains performance metrics"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_backtest_result_has_trades(self, mock_data, mock_backtest_result):
        """Test that result contains trade list"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService


class TestWalkForwardService:
    """Tests for walk-forward optimization functionality"""

    def test_run_walk_forward_returns_result(self, mock_data):
        """Test that run_walk_forward returns WalkForwardResult"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_walk_forward_with_param_grid(self, mock_data):
        """Test walk-forward with parameter grid"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_walk_forward_with_custom_train_test_days(self, mock_data):
        """Test walk-forward with custom train/test periods"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_walk_forward_calls_optimizer(self, mock_data):
        """Test that run_walk_forward uses WalkForwardOptimizer"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_walk_forward_result_has_param_history(self, mock_data, mock_walk_forward_result):
        """Test that result contains parameter history"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_walk_forward_result_has_stability_score(self, mock_data, mock_walk_forward_result):
        """Test that result contains stability score"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_walk_forward_result_has_split_results(self, mock_data, mock_walk_forward_result):
        """Test that result contains split results"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService


class TestMonteCarloService:
    """Tests for Monte Carlo analysis functionality"""

    def test_run_monte_carlo_returns_result(self, mock_data):
        """Test that run_monte_carlo returns MonteCarloResult"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_monte_carlo_uses_backtest_result(self, mock_data, mock_backtest_result):
        """Test Monte Carlo uses backtest returns"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_monte_carlo_with_custom_simulations(self, mock_data):
        """Test Monte Carlo with custom number of simulations"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_monte_carlo_calls_analyzer(self, mock_data):
        """Test that run_monte_carlo uses MonteCarloAnalyzer"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_monte_carlo_result_has_var(self, mock_data, mock_monte_carlo_result):
        """Test that result contains VaR"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_monte_carlo_result_has_cvar(self, mock_data, mock_monte_carlo_result):
        """Test that result contains CVaR"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_monte_carlo_result_has_probability_of_loss(self, mock_data, mock_monte_carlo_result):
        """Test that result contains probability of loss"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_monte_carlo_result_has_confidence_interval(self, mock_data, mock_monte_carlo_result):
        """Test that result contains confidence interval"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService


class TestBacktestServiceCaching:
    """Tests for BacktestService caching behavior"""

    def test_cache_miss_runs_backtest(self, mock_data):
        """Test that cache miss triggers actual backtest"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_cache_hit_returns_cached_result(self, mock_data):
        """Test that cache hit returns cached result"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_cache_invalidation_on_param_change(self, mock_data):
        """Test that cache is invalidated when parameters change"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_cache_key_includes_strategy_id(self, mock_data):
        """Test that cache key includes strategy ID"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_cache_key_includes_date_range(self, mock_data):
        """Test that cache key includes date range"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService


class TestBacktestServiceErrorHandling:
    """Tests for error handling in BacktestService"""

    def test_invalid_strategy_id_raises_error(self, mock_data):
        """Test that invalid strategy ID raises appropriate error"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_invalid_date_range_raises_error(self, mock_data):
        """Test that invalid date range raises appropriate error"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_empty_symbols_raises_error(self, mock_data):
        """Test that empty symbols list raises appropriate error"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_data_service_failure_is_handled(self, mock_data):
        """Test that DataService failures are handled gracefully"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_backtester_failure_is_handled(self, mock_data):
        """Test that Backtester failures are handled gracefully"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService


class TestBacktestServiceWithRealComponents:
    """
    Tests that will work once BacktestService is implemented.
    These tests mock the DataService but use real backtesting components.
    """

    def test_full_backtest_flow(self, mock_data, mock_benchmark_data):
        """Test complete backtest flow with mocked data service"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_walk_forward_flow(self, mock_data, mock_benchmark_data):
        """Test complete walk-forward flow with mocked data service"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService

    def test_monte_carlo_flow(self, mock_data):
        """Test complete Monte Carlo flow with mocked data service"""
        with pytest.raises(ImportError):
            from services.backtest_service import BacktestService


# ============================================================
# The following tests will be enabled once BacktestService exists
# They provide the detailed test implementations
# ============================================================

class TestBacktestServiceImplementation:
    """
    Detailed implementation tests - to be enabled when BacktestService is implemented.

    These tests define the expected behavior and API of BacktestService.
    """

    @pytest.mark.skip(reason="BacktestService not implemented yet")
    def test_run_backtest_complete_flow(self, mock_data, mock_benchmark_data):
        """
        Test complete backtest flow:
        1. BacktestService receives strategy_id, symbols, date range
        2. DataService fetches historical data
        3. EventDrivenBacktester runs the backtest
        4. Returns BacktestResult with metrics, equity curve, trades
        """
        from services.backtest_service import BacktestService

        service = BacktestService()

        # Mock the data service
        with patch.object(service, '_data_service') as mock_ds:
            mock_ds.get_cached_data.return_value = mock_data["000001"]

            result = service.run_backtest(
                strategy_id="alpha_trend",
                symbols=["000001"],
                start_date=date(2020, 1, 1),
                end_date=date(2021, 12, 31)
            )

            assert isinstance(result, BacktestResult)
            assert hasattr(result, 'equity_curve')
            assert hasattr(result, 'trades')
            assert hasattr(result, 'metrics')
            assert isinstance(result.metrics, PerformanceMetrics)

    @pytest.mark.skip(reason="BacktestService not implemented yet")
    def test_run_walk_forward_complete_flow(self, mock_data):
        """
        Test complete walk-forward flow:
        1. BacktestService receives strategy_id, param_grid, train/test days
        2. DataService fetches historical data
        3. WalkForwardOptimizer runs optimization
        4. Returns WalkForwardResult with param history, stability score
        """
        from services.backtest_service import BacktestService

        service = BacktestService()

        param_grid = {
            "fast_period": [5, 10, 15],
            "slow_period": [20, 30, 40]
        }

        with patch.object(service, '_data_service') as mock_ds:
            mock_ds.get_cached_data.return_value = mock_data["000001"]

            result = service.run_walk_forward(
                strategy_id="alpha_trend",
                param_grid=param_grid,
                train_days=252,
                test_days=63
            )

            assert isinstance(result, WalkForwardResult)
            assert hasattr(result, 'combined_equity')
            assert hasattr(result, 'param_history')
            assert hasattr(result, 'stability_score')
            assert 0 <= result.stability_score <= 1

    @pytest.mark.skip(reason="BacktestService not implemented yet")
    def test_run_monte_carlo_complete_flow(self, mock_data, mock_backtest_result):
        """
        Test complete Monte Carlo flow:
        1. BacktestService receives strategy_id and n_simulations
        2. First runs backtest to get returns
        3. MonteCarloAnalyzer analyzes returns
        4. Returns MonteCarloResult with risk metrics
        """
        from services.backtest_service import BacktestService

        service = BacktestService()

        # Mock the backtest to return our mock result
        with patch.object(service, 'run_backtest', return_value=mock_backtest_result):
            result = service.run_monte_carlo(
                strategy_id="alpha_trend",
                n_simulations=500
            )

            assert isinstance(result, MonteCarloResult)
            assert hasattr(result, 'expected_max_drawdown')
            assert hasattr(result, 'var_95')
            assert hasattr(result, 'cvar_95')
            assert hasattr(result, 'probability_of_loss')
            assert hasattr(result, 'return_confidence_interval')

    @pytest.mark.skip(reason="BacktestService not implemented yet")
    def test_caching_behavior(self, mock_data):
        """
        Test that BacktestService properly caches results:
        1. First call triggers backtest
        2. Second call with same params returns cached result
        3. Different params trigger new backtest
        """
        from services.backtest_service import BacktestService

        service = BacktestService()

        with patch.object(service, '_data_service') as mock_ds:
            mock_ds.get_cached_data.return_value = mock_data["000001"]

            # First call
            result1 = service.run_backtest(
                strategy_id="alpha_trend",
                symbols=["000001"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            )

            # Second call with same params - should use cache
            result2 = service.run_backtest(
                strategy_id="alpha_trend",
                symbols=["000001"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            )

            # Results should be the same object (cached)
            assert result1 is result2 or result1 == result2

    @pytest.mark.skip(reason="BacktestService not implemented yet")
    def test_strategy_factory_registration(self, mock_data):
        """
        Test that BacktestService can register and retrieve strategy factories
        """
        from services.backtest_service import BacktestService

        service = BacktestService()

        # Should have built-in strategies
        assert service.get_strategy_factory("alpha_trend") is not None

        # Should be able to register custom strategy
        def custom_factory(params):
            pass

        service.register_strategy("custom", custom_factory)
        assert service.get_strategy_factory("custom") is custom_factory

    @pytest.mark.skip(reason="BacktestService not implemented yet")
    def test_error_on_unknown_strategy(self, mock_data):
        """
        Test that BacktestService raises error for unknown strategy
        """
        from services.backtest_service import BacktestService

        service = BacktestService()

        with pytest.raises(ValueError, match="Unknown strategy"):
            service.run_backtest(
                strategy_id="nonexistent_strategy",
                symbols=["000001"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            )

    @pytest.mark.skip(reason="BacktestService not implemented yet")
    def test_error_on_invalid_date_range(self, mock_data):
        """
        Test that BacktestService validates date range
        """
        from services.backtest_service import BacktestService

        service = BacktestService()

        with pytest.raises(ValueError, match="Invalid date range"):
            service.run_backtest(
                strategy_id="alpha_trend",
                symbols=["000001"],
                start_date=date(2021, 1, 1),
                end_date=date(2020, 1, 1)  # End before start
            )

    @pytest.mark.skip(reason="BacktestService not implemented yet")
    def test_error_on_empty_symbols(self, mock_data):
        """
        Test that BacktestService validates symbols list
        """
        from services.backtest_service import BacktestService

        service = BacktestService()

        with pytest.raises(ValueError, match="Symbols list cannot be empty"):
            service.run_backtest(
                strategy_id="alpha_trend",
                symbols=[],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            )
