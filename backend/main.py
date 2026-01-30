from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import pandas as pd
import json
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

# Import Engine - try new modular engine first, fall back to legacy
try:
    from engine.backtester import EventDrivenBacktester as Backtester, Strategy
    from engine.execution import CompositeExecutionModel as SimpleExecutionModel
    from engine.models import Order, OrderSide, OrderType, Fill
    NEW_ENGINE = True
except ImportError:
    # Legacy imports
    from engine.backtester import Backtester, SimpleExecutionModel
    NEW_ENGINE = False

from engine.data_loader import generate_mock_data, fetch_stock_data

# Import new modular engine components
try:
    from services.backtest_service import BacktestService
    from engine.backtester import EventDrivenBacktester
    from engine.walk_forward import WalkForwardOptimizer
    from engine.monte_carlo import MonteCarloAnalyzer
    ADVANCED_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced engine components not available: {e}")
    ADVANCED_ENGINE_AVAILABLE = False

# Import new modules
from strategies.registry import (
    get_strategy_info,
    get_all_strategies,
    create_strategy,
    get_strategy_list_for_api,
    get_stock_pool_for_strategy
)
from services.data_service import get_data_service
from services.signal_service import get_signal_service
from database.connection import init_db, test_connection
from database.repository import MarketStatusRepository
from scheduler import start_scheduler, get_job_status, trigger_job, get_realtime_prices, get_realtime_price
from services.cache_service import get_cache_service

app = FastAPI(title="Strategy Supermarket API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IN-MEMORY CACHE FOR DEMO ---
# In a real app, this would be a database or Redis
STRATEGY_RESULTS = {}

# --- Initialize BacktestService ---
backtest_service = None
if ADVANCED_ENGINE_AVAILABLE:
    try:
        backtest_service = BacktestService()
        logger.info("BacktestService initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize BacktestService: {e}")

# --- STARTUP EVENTS ---

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("Starting Strategy Supermarket API...")

    # Test database connection (optional - fails if MySQL not configured)
    try:
        test_connection()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"⚠ Database connection failed: {e}")
        print("  Running in mock data mode")

    # Start background scheduler
    try:
        start_scheduler()
        print("✓ Background scheduler started")
    except Exception as e:
        print(f"⚠ Scheduler start failed: {e}")

    # Register strategies with BacktestService
    if backtest_service is not None:
        try:
            from strategies.alpha_trend import AlphaTrendStrategy
            backtest_service.register_strategy("alpha_trend", AlphaTrendStrategy)
            print("✓ Registered alpha_trend strategy with BacktestService")

            # Register other available strategies
            try:
                from strategies.momentum import MomentumStrategy
                backtest_service.register_strategy("momentum", MomentumStrategy)
                print("✓ Registered momentum strategy with BacktestService")
            except ImportError:
                pass

            try:
                from strategies.mean_reversion import MeanReversionStrategy
                backtest_service.register_strategy("mean_reversion", MeanReversionStrategy)
                print("✓ Registered mean_reversion strategy with BacktestService")
            except ImportError:
                pass

            try:
                from strategies.sector_rotation import SectorRotationStrategy
                backtest_service.register_strategy("sector_rotation", SectorRotationStrategy)
                print("✓ Registered sector_rotation strategy with BacktestService")
            except ImportError:
                pass

            print("✓ Strategies registered with BacktestService")
        except Exception as e:
            print(f"⚠ Strategy registration failed: {e}")


# --- BACKTEST HELPERS ---

def run_backtest(strategy_id: str, symbols: List[str] = None, use_real_data: bool = True):
    """
    Run backtest for a strategy and cache results.

    Note: Always uses real data from database. If no data is available,
    returns an empty result instead of using mock data.
    """
    # Check database cache first
    from database.connection import get_session
    from database.models import StrategyBacktest
    from sqlalchemy import and_
    import json
    from io import StringIO

    # Get strategy info
    strategy_info = get_strategy_info(strategy_id)
    if not strategy_info:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Get symbols to trade
    if symbols is None:
        symbols = strategy_info.default_symbols

    # Check cache in database
    # Use 10 years of data
    start_dt = (datetime.now() - pd.Timedelta(days=365*10)).date()
    end_dt = datetime.now().date()

    with get_session() as session:
        cached = session.query(StrategyBacktest).filter(
            and_(
                StrategyBacktest.strategy_id == strategy_id,
                StrategyBacktest.start_date == start_dt,
                StrategyBacktest.end_date == end_dt
            )
        ).first()

        if cached and cached.last_updated:
            # Check if cache is recent (within 1 day)
            cache_age = (datetime.now() - cached.last_updated).total_seconds() / 3600
            if cache_age < 24:
                logger.info(f"Using cached backtest for {strategy_id}")
                try:
                    # Parse the JSON records
                    equity_records = json.loads(cached.equity_curve)
                    # Convert to DataFrame
                    equity_df = pd.DataFrame(equity_records)
                    # Set index from timestamp
                    if 'timestamp' in equity_df.columns:
                        equity_df.set_index('timestamp', inplace=True)
                        equity_df.index = pd.to_datetime(equity_df.index)
                    return {
                        "equity": equity_df,
                        "metrics": json.loads(cached.metrics),
                        "trades": json.loads(cached.trades) if cached.trades else []
                    }
                except Exception as e:
                    logger.warning(f"Failed to read cache: {e}, running new backtest")

    # Create strategy instance
    strategy = create_strategy(strategy_id)

    execution_model = SimpleExecutionModel()
    backtester = Backtester(strategy, execution_model, initial_capital=Decimal("100000"))

    # Load Data from database (NO MOCK DATA FALLBACK)
    data_service = get_data_service()

    # Get data for first symbol only (backtester is single-symbol)
    symbol = symbols[0] if symbols else "DEMO"
    data = data_service.get_cached_data([symbol], start_dt, end_dt)

    if data.empty:
        # No data available - return empty result (not mock data)
        logger.warning(f"No data available for {symbol}")
        # Return empty dataframe with proper structure
        empty_df = pd.DataFrame({
            'equity': [100000.0],
            'returns': [0.0]
        })
        empty_df.index = pd.to_datetime([datetime.now()])
        return {
            "equity": empty_df,
            "metrics": {
                "sharpe": 0, "calmar": 0, "pl_ratio": 0,
                "avg_hold_days": 0, "strategy_return": 0,
                "sortino": 0, "alpha": 0, "beta": 1.0,
                "benchmark_return": 0, "win_count": 0,
                "loss_count": 0, "volatility": 0,
                "excess_max_drawdown": 0, "ytd_return": 0,
                "mtd_return": 0, "consecutive_wins": 0,
                "drawdown_period": "N/A",
                "cagr": 0, "max_drawdown": 0,
                "win_rate": 0, "total_return": 0
            },
            "trades": [],
            "backtester": backtester
        }

    # Filter to single symbol and ensure format
    if "symbol" in data.columns:
        data = data[data["symbol"] == symbol].copy()

    # Convert to dict format for new EventDrivenBacktester
    # The new backtester expects Dict[str, pd.DataFrame]
    data_dict = {symbol: data}

    # Run backtest
    result = backtester.run(data_dict)
    equity_df = result.equity_curve if hasattr(result, 'equity_curve') else result
    metrics = _calculate_metrics(equity_df["returns"])

    # Save to database cache
    try:
        with get_session() as session:
            # Check if exists
            existing = session.query(StrategyBacktest).filter(
                and_(
                    StrategyBacktest.strategy_id == strategy_id,
                    StrategyBacktest.start_date == start_dt,
                    StrategyBacktest.end_date == end_dt
                )
            ).first()

            # Create records with timestamp included
            equity_records = [
                {
                    "timestamp": str(idx),  # Include timestamp in each record
                    "equity": float(row["equity"]),
                    "returns": float(row["returns"]) if pd.notna(row["returns"]) else 0.0
                }
                for idx, row in equity_df.iterrows()
            ]
            equity_json = json.dumps(equity_records)
            # Get trades from result (new format returns BacktestResult with trades list)
            trades_list = []
            if hasattr(result, 'trades'):
                trades_list = [
                    {"timestamp": str(t.timestamp), "symbol": t.order.symbol, "side": t.order.side.value}
                    for t in result.trades
                ]

            if existing:
                existing.equity_curve = equity_json
                existing.metrics = json.dumps(metrics)
                existing.trades = json.dumps(trades_list)
                existing.last_updated = datetime.now()
            else:
                record = StrategyBacktest(
                    strategy_id=strategy_id,
                    symbols=json.dumps(symbols),
                    start_date=start_dt,
                    end_date=end_dt,
                    equity_curve=equity_json,
                    metrics=json.dumps(metrics),
                    trades=json.dumps(trades_list)
                )
                session.add(record)

            session.commit()
            logger.info(f"Saved backtest cache for {strategy_id}")
    except Exception as e:
        logger.error(f"Failed to save backtest cache: {e}")

    # Cache Result in memory
    result = {
        "backtester": backtester,
        "equity": equity_df,
        "metrics": metrics
    }
    STRATEGY_RESULTS[strategy_id] = result
    return result


def _calculate_metrics(returns: pd.Series):
    """Calculate comprehensive performance metrics."""
    # Handle empty or all-zero returns
    if returns.empty or (returns == 0).all():
        return {
            "sharpe": 0, "calmar": 0, "pl_ratio": 0, "avg_hold_days": 0,
            "strategy_return": 0, "sortino": 0, "alpha": 0, "beta": 1.0,
            "benchmark_return": 0.05, "win_count": 0, "loss_count": 0,
            "volatility": 0, "excess_max_drawdown": 0, "ytd_return": 0,
            "mtd_return": 0, "consecutive_wins": 0, "drawdown_period": "N/A",
            "cagr": 0, "max_drawdown": 0, "win_rate": 0, "total_return": 0
        }

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    vol = returns.std() * (252 ** 0.5)
    sharpe = ann_return / vol if vol != 0 else 0

    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Win Rate
    wins = len(returns[returns > 0])
    total = len(returns[returns != 0])
    win_rate = wins / total if total > 0 else 0

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * (252 ** 0.5) if len(downside_returns) > 0 else 0.0001
    sortino = ann_return / downside_std if downside_std > 0 else 0

    # Calmar Ratio (CAGR / AbsMaxDD)
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # P/L Ratio (average win / average loss)
    avg_win = returns[returns > 0].mean() if wins > 0 else 0
    losses = returns[returns < 0]
    avg_loss = abs(losses.mean()) if len(losses) > 0 and not pd.isna(losses.mean()) else 0.0001
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Alpha and Beta (vs benchmark, assuming 0 benchmark return for simplicity)
    # In real implementation, would compare against index
    alpha = ann_return  # Simplified - excess return over risk-free
    beta = 1.0  # Simplified - would calculate from covariance

    # Volatility (annualized)
    volatility = vol

    # Win/Loss count
    win_count = wins
    loss_count = total - wins

    # Consecutive wins
    max_consecutive_wins = 0
    current_streak = 0
    for ret in returns:
        if ret > 0:
            current_streak += 1
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        else:
            current_streak = 0

    # Calculate drawdown period (date range of max drawdown)
    max_dd_idx_val = drawdown.idxmin()  # This is the index value (Timestamp), not position
    max_dd_date = max_dd_idx_val  # The index value itself is the date

    # YTD and MTD returns
    current_date = returns.index[-1]
    ytd_start = current_date.replace(month=1, day=1)
    ytd_returns = returns[returns.index >= ytd_start]
    ytd_return = (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0

    mtd_start = current_date.replace(day=1)
    mtd_returns = returns[returns.index >= mtd_start]
    mtd_return = (1 + mtd_returns).prod() - 1 if len(mtd_returns) > 0 else 0

    # Format drawdown period safely
    try:
        dd_period_str = max_dd_date.strftime("%Y/%m") if pd.notna(max_dd_date) else "N/A"
    except:
        dd_period_str = "N/A"

    return {
        # Row 1 (Hero)
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "pl_ratio": float(pl_ratio),
        "avg_hold_days": 17.8,  # Would be calculated from actual trades

        # Row 2 (Returns)
        "strategy_return": float(ann_return),  # Strategy period return
        "sortino": float(sortino),
        "alpha": float(alpha),
        "beta": float(beta),

        # Row 3 (Risk)
        "benchmark_return": 0.05,  # Would be actual benchmark return
        "win_count": int(win_count),
        "loss_count": int(loss_count),
        "volatility": float(volatility),
        "excess_max_drawdown": float(max_drawdown),  # Using same as max drawdown

        # Row 4 (Time)
        "ytd_return": float(ytd_return),
        "mtd_return": float(mtd_return),
        "consecutive_wins": int(max_consecutive_wins),
        "drawdown_period": dd_period_str,

        # Original metrics (for backward compatibility)
        "cagr": float(ann_return),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "total_return": float(total_return)
    }


# --- ENDPOINTS ---

# Simple in-memory cache for strategy list (1 minute TTL)
_strategies_list_cache = {"data": None, "time": None}
_STRATEGIES_CACHE_TTL = 60  # seconds


@app.get("/api/strategies")
def get_strategies(
    search: Optional[str] = None,
    sort: Optional[str] = None,
    order: Optional[str] = "desc"
):
    """
    Get list of all strategies with optional search and sort.

    - search: Filter by name/description
    - sort: Sort field (cagr, sharpe, max_drawdown, latest_signal)
    - order: asc or desc
    """
    import time
    current_time = time.time()

    # Check cache
    if _strategies_list_cache["data"] and (current_time - _strategies_list_cache["time"]) < _STRATEGIES_CACHE_TTL:
        results = _strategies_list_cache["data"]
    else:
        # Build strategy list
        strategy_list = get_all_strategies()
        results = []

        for info in strategy_list:
            # Run backtest to get metrics (use cached results when possible)
            try:
                res = run_backtest(info.strategy_id, use_real_data=True)
                metrics = res["metrics"]
                equity_df = res["equity"]

                # Get sparkline data (last 20 points)
                sparkline_data = equity_df["equity"].iloc[-20:].tolist() if not equity_df.empty else []

                # Get latest signal info
                signal_service = get_signal_service()
                has_recent_signal = signal_service.is_signal_recent(info.strategy_id, minutes=15)

                # Calculate additional metrics
                total_trades = len(equity_df) if not equity_df.empty else 0
                avg_hold_days = 17.8  # Default, would be calculated from actual trades

                result_item = {
                    "id": info.strategy_id,
                    "name": info.name,
                    "description": info.description,
                    "tags": info.tags,
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "win_rate": metrics["win_rate"],
                    "sparkline": sparkline_data,
                    "latest_signal": {
                        "has_recent": has_recent_signal,
                        "time": "今天 09:30" if has_recent_signal else None
                    },
                    "total_trades": total_trades,
                    "avg_hold_days": avg_hold_days,
                    "is_active": info.is_active
                }

                results.append(result_item)

            except Exception as e:
                logger.error(f"Error processing {info.strategy_id}: {e}")
                continue

        # Update cache
        _strategies_list_cache["data"] = results
        _strategies_list_cache["time"] = current_time

    # Apply search filter
    if search:
        search_lower = search.lower()
        results = [
            r for r in results
            if (search_lower in r["name"].lower() or
                search_lower in r["description"].lower() or
                any(search_lower in tag.lower() for tag in r["tags"]))
        ]

    # Apply sorting
    if sort:
        reverse = order.lower() == "desc"
        results = sorted(results, key=lambda x: x.get(sort, 0), reverse=reverse)

    return results


@app.get("/api/strategies/{id}/info")
def get_strategy_info_endpoint(id: str):
    """Get strategy information (name, description, tags, etc.)."""
    strategy_info = get_strategy_info(id)
    if not strategy_info:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Get overall metrics for the header (not year-specific)
    res = run_backtest(id, use_real_data=False)
    metrics = res["metrics"]

    return {
        "id": strategy_info.strategy_id,
        "name": strategy_info.name,
        "description": strategy_info.description,
        "tags": strategy_info.tags,
        "is_active": strategy_info.is_active,
        "total_metrics": {
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "total_return": metrics["total_return"]
        }
    }


@app.get("/api/strategies/{id}/yearly-data")
def get_yearly_data(id: str):
    """Get yearly performance data for the sidebar based on actual backtest results."""
    try:
        # Run backtest to get equity data
        res = run_backtest(id, use_real_data=True)
        equity_df = res["equity"]

        if equity_df.empty:
            return []

        # Add year column - convert to native Python type
        equity_df_with_year = equity_df.copy()
        equity_df_with_year['year'] = equity_df_with_year.index.year.astype(int)

        # Get unique years from the actual equity data
        actual_years = sorted(equity_df_with_year['year'].unique(), reverse=True)

        # Calculate yearly returns
        yearly_data = []
        current_year = datetime.now().year

        for year in actual_years:
            year_equity = equity_df_with_year[equity_df_with_year['year'] == year]
            if len(year_equity) > 1:
                start_val = float(year_equity["equity"].iloc[0])
                end_val = float(year_equity["equity"].iloc[-1])
                year_return = (end_val - start_val) / start_val if start_val > 0 else 0.0
            else:
                year_return = 0.0

            yearly_data.append({
                "year": int(year),
                "ret": float(year_return),
                "is_running": bool(year == current_year)
            })

        return yearly_data

    except Exception as e:
        logger.error(f"Error getting yearly data: {e}")
        import traceback
        traceback.print_exc()
        return []


@app.get("/api/strategies/{id}/metrics")
def get_metrics(id: str, year: Optional[int] = None):
    """Get performance metrics for a strategy, optionally filtered by year."""
    import math
    cache = get_cache_service()

    # Try cache first
    cached = cache.get(id, "metrics", year)
    if cached:
        return cached

    res = run_backtest(id, use_real_data=True)
    df = res["equity"]

    if year and not df.empty:
        # Filter by year and recalculate metrics
        df_filtered = df[df.index.year == year]
        if not df_filtered.empty:
            metrics = _calculate_metrics(df_filtered["returns"])
        else:
            # No data for this year
            metrics = res["metrics"]
    else:
        metrics = res["metrics"]

    # Clean NaN/Inf values for JSON serialization
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            metrics[key] = 0

    # Cache the result
    cache.set(id, "metrics", metrics, year)
    return metrics


@app.get("/api/strategies/{id}/detailed-metrics")
def get_detailed_metrics(id: str):
    """Get detailed performance metrics organized by category for Data Metrics tab."""
    res = run_backtest(id, use_real_data=True)
    m = res["metrics"]

    # Handle missing keys from old cached data
    def get_metric(key, default=0):
        return m.get(key, default)

    return {
        # Row 1 (Hero) - Professional Metrics
        "sharpe_ratio": {
            "value": float(get_metric("sharpe")),
            "desc": "性价比之王"
        },
        "calmar_ratio": {
            "value": float(get_metric("calmar")),
            "desc": "回撤恢复力"
        },
        "pl_ratio": {
            "value": float(get_metric("pl_ratio")),
            "desc": "赚赔力度"
        },
        "avg_hold_days": {
            "value": get_metric("avg_hold_days", 17.8),
            "desc": "策略频率"
        },

        # Row 2 (Returns)
        "strategy_return": {
            "value": float(get_metric("strategy_return", m.get("cagr", 0))) * 100,  # Convert to percentage
            "desc": "策略区间收益"
        },
        "sortino_ratio": {
            "value": float(get_metric("sortino")),
            "desc": "索提诺比率"
        },
        "alpha": {
            "value": float(get_metric("alpha")),
            "desc": "Alpha (α)"
        },
        "beta": {
            "value": float(get_metric("beta", 1.0)),
            "desc": "Beta (β)"
        },

        # Row 3 (Risk)
        "benchmark_return": {
            "value": float(get_metric("benchmark_return", 0.05)) * 100,
            "desc": "基准收益"
        },
        "win_loss_count": {
            "value": f"{get_metric('win_count', 0)}胜 / {get_metric('loss_count', 0)}负",
            "desc": "胜负场"
        },
        "volatility": {
            "value": float(get_metric("volatility", 0.15)) * 100,
            "desc": "波动率"
        },
        "excess_max_drawdown": {
            "value": float(get_metric("excess_max_drawdown", m.get("max_drawdown", 0))) * 100,
            "desc": "超额最大回撤"
        },

        # Row 4 (Time)
        "ytd_return": {
            "value": float(get_metric("ytd_return", 0)) * 100,
            "desc": "今年收益 (YTD)"
        },
        "mtd_return": {
            "value": float(get_metric("mtd_return", 0)) * 100,
            "desc": "本月收益 (MTD)"
        },
        "consecutive_wins": {
            "value": f"{get_metric('consecutive_wins', 0)}天",
            "desc": "连红天数"
        },
        "drawdown_period": {
            "value": get_metric("drawdown_period", "N/A"),
            "desc": "回撤区间"
        }
    }


@app.get("/api/strategies/{id}/walk-forward")
def get_walk_forward_analysis(
    id: str,
    train_days: int = Query(252, description="Training period days"),
    test_days: int = Query(63, description="Test period days")
):
    """Run Walk-Forward optimization analysis."""
    if not ADVANCED_ENGINE_AVAILABLE or backtest_service is None:
        raise HTTPException(
            status_code=503,
            detail="Walk-Forward analysis not available. Advanced engine components not initialized."
        )

    try:
        # Verify strategy is registered
        if backtest_service.get_strategy_factory(id) is None:
            # Try to register dynamically based on strategy_id
            strategy_info = get_strategy_info(id)
            if not strategy_info:
                raise HTTPException(status_code=404, detail=f"Strategy '{id}' not found")
            raise HTTPException(
                status_code=400,
                detail=f"Strategy '{id}' not registered with BacktestService"
            )

        # Define parameter grid based on strategy
        param_grid = {
            "short_window": [5, 10, 20],
            "long_window": [20, 40, 60]
        }

        result = backtest_service.run_walk_forward(
            strategy_id=id,
            param_grid=param_grid,
            train_days=train_days,
            test_days=test_days
        )

        return {
            "stability_score": result.stability_score,
            "param_history": result.param_history,
            "split_count": len(result.split_results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Walk-Forward error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies/{id}/monte-carlo")
def get_monte_carlo_analysis(
    id: str,
    n_simulations: int = Query(1000, description="Number of simulations")
):
    """Run Monte Carlo risk analysis."""
    if not ADVANCED_ENGINE_AVAILABLE or backtest_service is None:
        raise HTTPException(
            status_code=503,
            detail="Monte Carlo analysis not available. Advanced engine components not initialized."
        )

    try:
        # Verify strategy is registered
        if backtest_service.get_strategy_factory(id) is None:
            strategy_info = get_strategy_info(id)
            if not strategy_info:
                raise HTTPException(status_code=404, detail=f"Strategy '{id}' not found")
            raise HTTPException(
                status_code=400,
                detail=f"Strategy '{id}' not registered with BacktestService"
            )

        result = backtest_service.run_monte_carlo(
            strategy_id=id,
            n_simulations=n_simulations
        )

        return {
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
            "expected_max_drawdown": result.expected_max_drawdown,
            "probability_of_loss": result.probability_of_loss,
            "confidence_interval": {
                "lower": result.return_confidence_interval[0],
                "upper": result.return_confidence_interval[1]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Monte Carlo error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies/{id}/drawdown")
def get_drawdown(id: str, year: Optional[int] = None):
    """Get drawdown data for underwater chart, optionally filtered by year."""
    cache = get_cache_service()

    # Try cache first
    cached = cache.get(id, "drawdown", year)
    if cached:
        return cached

    try:
        res = run_backtest(id, use_real_data=True)
        df = res["equity"]

        if df.empty:
            return []

        if year:
            df = df[df.index.year == year]
            if df.empty:
                return []

        # Fill NaN values in returns
        returns = df["returns"].fillna(0)

        # Calculate drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max

        # Handle any remaining NaN or inf values
        drawdown = drawdown.fillna(0).replace([float('inf'), float('-inf')], 0)

        result = [
            {"date": str(ts)[:10], "drawdown": float(dd) * 100}  # Convert to string date, percentage
            for ts, dd in zip(df.index, drawdown)
        ]

        # Cache the result
        cache.set(id, "drawdown", result, year)
        return result
    except Exception as e:
        logger.error(f"Error calculating drawdown: {e}")
        import traceback
        traceback.print_exc()
        return []


@app.get("/api/strategies/{id}/monthly-returns")
def get_monthly_returns(id: str, year: Optional[int] = None):
    """Get monthly returns for heatmap display, optionally filtered by year."""
    cache = get_cache_service()

    # Try cache first
    cached = cache.get(id, "monthly", year)
    if cached:
        return cached

    res = run_backtest(id, use_real_data=True)
    df = res["equity"]

    if df.empty:
        return []

    # Add year and month columns
    df_with_ym = df.copy()
    df_with_ym['year'] = df_with_ym.index.year
    df_with_ym['month'] = df_with_ym.index.month

    # Filter by year if specified
    years_to_process = [year] if year else sorted(df_with_ym['year'].unique())

    # Calculate monthly returns
    monthly_data = []
    for y in years_to_process:
        for month in range(1, 13):
            month_data = df_with_ym[(df_with_ym['year'] == y) & (df_with_ym['month'] == month)]
            if len(month_data) > 0:
                start_val = float(month_data["equity"].iloc[0])
                end_val = float(month_data["equity"].iloc[-1])
                monthly_return = (end_val - start_val) / start_val if start_val > 0 else 0
                monthly_data.append({
                    "year": int(y),
                    "month": month,
                    "return": float(monthly_return) * 100  # Convert to percentage
                })

    # Cache the result
    cache.set(id, "monthly", monthly_data, year)
    return monthly_data


@app.get("/api/strategies/{id}/equity_curve")
def get_equity_curve(id: str, year: Optional[int] = None):
    """Get equity curve data for a strategy, optionally filtered by year."""
    cache = get_cache_service()

    # Try cache first
    cached = cache.get(id, "equity", year)
    if cached:
        return cached

    res = run_backtest(id, use_real_data=True)
    df = res["equity"]

    if year and not df.empty:
        df = df[df.index.year == year]

    if df.empty:
        return []

    result = [
        {"date": ts.strftime("%Y-%m-%d"), "value": float(val), "benchmark": float(val) * 0.9}
        for ts, val in zip(df.index, df["equity"])
    ]

    # Cache the result
    cache.set(id, "equity", result, year)
    return result


@app.get("/api/strategies/{id}/transactions")
def get_transactions(id: str, is_subscribed: bool = False, year: Optional[int] = None):
    """Get transaction history for a strategy with encryption for non-subscribers."""
    try:
        # Get stock pool for sector mapping
        from database.repository import StockPoolRepository
        stock_pool_repo = StockPoolRepository()
        stock_pool = stock_pool_repo.get_stock_pool()
        sector_map = {s['symbol']: s.get('sector', '其他') for s in stock_pool}

        # Run backtest to get trades
        res = run_backtest(id, use_real_data=True)
        trades = res["backtester"].trades

        # Filter by year if specified
        if year:
            trades = [t for t in trades if t.timestamp.year == year]

        api_trades = []

        for i, trade in enumerate(reversed(trades)):
            # Determine if this trade should be encrypted (recent trades = encrypted)
            # For demo: last 3 trades are considered "active"
            is_encrypted = (i < 3) and (not is_subscribed)

            symbol = trade.order.symbol
            sector = sector_map.get(symbol, '其他')

            trade_data = {
                "date": trade.timestamp.strftime("%Y-%m-%d"),
                "time": trade.timestamp.strftime("%H:%M:%S"),
                "symbol": symbol,
                "side": trade.order.side.value,
                "price": float(trade.fill_price),
                "quantity": float(trade.fill_quantity),
                "pnl": float(trade.fill_quantity * trade.fill_price * 0.01),  # Simplified P&L
                "is_encrypted": is_encrypted
            }

            if is_encrypted:
                # Mask data for non-subscribers (PRD requirement)
                trade_data["symbol"] = sector  # Show sector instead of symbol
                trade_data["original_symbol"] = None
                trade_data["price"] = None  # Hide price
                trade_data["quantity"] = None  # Hide quantity
                # Blur time to hour range
                hour = trade.timestamp.hour
                time_range = f"{hour:02d}:00-{hour+1:02d}"
                trade_data["time"] = time_range
                # Show floating P&L as percentage
                trade_data["floating_pnl"] = 0.05 + (i * 0.02)  # Demo value
                trade_data["pnl_percent"] = trade_data["floating_pnl"]
            else:
                # Full data for subscribers or closed trades
                trade_data["sector"] = sector
                trade_data["original_symbol"] = symbol
                trade_data["floating_pnl"] = None
                trade_data["pnl_percent"] = trade_data["pnl"] * 100

            api_trades.append(trade_data)

        return api_trades

    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return []


@app.get("/api/strategies/{id}/holdings")
def get_holdings(id: str, is_subscribed: bool = False):
    """Get current holdings for a strategy with masking for non-subscribers."""
    try:
        # Try to get from database first (strategy_positions table)
        from database.repository import PositionRepository
        from database.models import StrategyPosition

        position_repo = PositionRepository()
        db_positions = position_repo.get_positions(id)

        holdings = []
        total_pnl = 0.0
        total_val = 100000.0  # Initial capital

        if db_positions:
            # Use database positions
            for pos in db_positions:
                holdings.append({
                    "symbol": pos.symbol,
                    "name": pos.symbol,  # Will be replaced by actual name if available
                    "sector": pos.sector or "其他",
                    "direction": pos.direction or "Long",
                    "days_held": pos.days_held,
                    "weight": f"{pos.weight}%" if pos.weight else "0%",
                    "entry_price": float(pos.entry_price) if pos.entry_price else 0,
                    "current_price": float(pos.current_price) if pos.current_price else 0,
                    "pnl": float(pos.floating_pnl) / 100 if pos.floating_pnl else 0  # Convert from % to decimal
                })
                total_pnl += float(pos.floating_pnl) if pos.floating_pnl else 0
        else:
            # Fallback to backtest data
            from database.repository import StockPoolRepository
            stock_pool_repo = StockPoolRepository()
            stock_pool = stock_pool_repo.get_stock_pool()
            sector_map = {s['symbol']: s.get('sector', '其他') for s in stock_pool}

            res = run_backtest(id, use_real_data=True)
            backtester = res["backtester"]

            for symbol, position in backtester.portfolio.positions.items():
                if position.quantity > 0:
                    entry_price = float(position.avg_cost) if position.avg_cost else 0
                    sector = sector_map.get(symbol, '其他')

                    current_price = entry_price * (1 + 0.02 * (hash(symbol) % 10 - 5))
                    current_value = position.quantity * current_price
                    pnl = current_value - (position.quantity * entry_price)

                    holdings.append({
                        "symbol": symbol,
                        "name": sector,
                        "sector": sector,
                        "direction": "Long",
                        "days_held": 12,
                        "weight": "25%",
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "pnl": pnl / (position.quantity * entry_price) if entry_price > 0 else 0
                    })

                    total_pnl += pnl

        # Calculate average P&L for total
        if holdings:
            total_pnl_pct = total_pnl / len(holdings) if db_positions else (total_pnl / total_val * 100 if total_val > 0 else 0)
        else:
            total_pnl_pct = 0.0

        # Format response based on subscription status
        response_list = []
        for h in holdings:
            item = {
                "sector": h["sector"],
                "direction": h.get("direction", "Long"),
                "days_held": h.get("days_held", 0),
                "weight": h.get("weight", "0%"),
                "pnl_pct": h.get("pnl", 0) * 100  # Convert to percentage
            }

            if is_subscribed:
                item["symbol"] = h["symbol"]
                item["name"] = h["name"]
                item["current_price"] = h.get("current_price")
            else:
                # Mask data for non-subscribers (PRD requirement)
                item["symbol"] = "HIDDEN"
                item["name"] = "HIDDEN"
                item["current_price"] = None

            response_list.append(item)

        return {
            "holdings": response_list,
            "total_pnl_pct": float(total_pnl_pct),
            "position_count": len(holdings)
        }

    except Exception as e:
        logger.error(f"Error getting holdings: {e}")
        import traceback
        traceback.print_exc()

        # Return empty holdings on error
        return {
            "holdings": [],
            "total_pnl_pct": 0.0,
            "position_count": 0
        }


@app.get("/api/strategies/{id}/signals")
def get_signals(id: str):
    """Get real-time signals for a strategy."""
    signal_service = get_signal_service()
    signals = signal_service.get_active_signals(id)

    return {
        "strategy_id": id,
        "signals": signals,
        "has_recent": signal_service.is_signal_recent(id)
    }


@app.get("/api/market/symbols")
def get_market_symbols(sector: Optional[str] = None):
    """Get available stock symbols."""
    data_service = get_data_service()

    try:
        pool_data = data_service.stock_pool_repo.get_stock_pool()

        if sector:
            pool_data = [s for s in pool_data if sector.lower() in s.get('sector', '').lower()]

        return pool_data[:100]  # Limit to 100

    except Exception as e:
        # Return fallback data
        return [
            {"symbol": "000001", "name": "平安银行", "sector": "金融"},
            {"symbol": "600519", "name": "贵州茅台", "sector": "消费"},
            {"symbol": "300750", "name": "宁德时代", "sector": "新能源"},
        ]


@app.get("/api/market/sectors")
def get_market_sectors():
    """Get available market sectors."""
    return [
        {"id": "finance", "name": "金融", "count": 45},
        {"id": "technology", "name": "科技", "count": 38},
        {"id": "healthcare", "name": "医药", "count": 25},
        {"id": "consumer", "name": "消费", "count": 32},
        {"id": "energy", "name": "能源", "count": 18},
        {"id": "materials", "name": "材料", "count": 22},
        {"id": "industrial", "name": "制造", "count": 28},
        {"id": "utilities", "name": "公用", "count": 12},
    ]


# --- ADMIN ENDPOINTS ---

@app.post("/api/admin/update-data")
def trigger_data_update(
    background_tasks: BackgroundTasks,
    force: bool = False,
    years: int = 1
):
    """
    Manually trigger stock data update.

    Args:
        force: Force update even if recent data exists (fetches all history)
        years: Number of years of historical data to fetch (default: 1)
    """
    background_tasks.add_task(_update_data_task, force, years)
    return {
        "status": "triggered",
        "message": f"Data update task started (force={force}, years={years})"
    }


def _update_data_task(force_update: bool = False, years: int = 1):
    """Background task for data update."""
    try:
        data_service = get_data_service()
        days_back = years * 365
        stats = data_service.update_stock_data(days_back=days_back, force_update=force_update)
        print(f"Data update completed: {stats}")
    except Exception as e:
        print(f"Data update failed: {e}")


@app.post("/api/admin/init-db")
def initialize_database():
    """Initialize database tables."""
    try:
        init_db(drop_tables=False)
        return {"status": "success", "message": "Database initialized"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/admin/jobs")
def get_scheduled_jobs():
    """Get status of scheduled jobs."""
    return get_job_status()


@app.post("/api/admin/jobs/{job_id}/trigger")
def trigger_scheduled_job(job_id: str):
    """Manually trigger a scheduled job."""
    success = trigger_job(job_id)
    if success:
        return {"status": "triggered", "job_id": job_id}
    else:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


# --- REALTIME PRICES ---


@app.get("/api/market/realtime")
def get_all_realtime_prices():
    """Get all cached realtime prices."""
    prices = get_realtime_prices()
    return {
        "count": len(prices),
        "timestamp": datetime.now().isoformat(),
        "prices": prices
    }


@app.get("/api/market/realtime/{symbol}")
def get_stock_realtime_price(symbol: str):
    """Get realtime price for a specific stock."""
    price_data = get_realtime_price(symbol)
    if price_data:
        return {
            "symbol": symbol,
            "data": price_data,
            "timestamp": price_data.get('timestamp', datetime.now()).isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"No realtime data for {symbol}")


# --- DATA MANAGEMENT ---

@app.post("/api/admin/fetch-data")
def trigger_data_fetch(background_tasks: BackgroundTasks, symbols: List[str] = None, days_back: int = 365):
    """
    Trigger manual data fetch from AkShare.

    This endpoint fetches stock data from AkShare and stores it in the database.
    Use this to initialize or update the stock data.

    Args:
        symbols: List of stock symbols to fetch (None = all CSI300 stocks)
        days_back: Number of days of historical data to fetch (default: 365 = 1 year)
    """
    def fetch_task():
        data_service = get_data_service()

        # Initialize stock pool if needed
        stock_pool = data_service.stock_pool_repo.get_stock_pool()
        if not stock_pool:
            data_service.init_stock_pool()

        # Fetch data
        stats = data_service.update_stock_data(symbols=symbols, days_back=days_back, force_update=True)

        logger.info(f"Data fetch complete: {stats}")

    background_tasks.add_task(fetch_task)

    return {
        "status": "started",
        "message": f"Data fetch initiated for {len(symbols) if symbols else 'CSI300'} symbols, {days_back} days back",
        "note": "This runs in the background. Check /api/admin/data-status for results."
    }


@app.get("/api/admin/data-status")
def get_data_status():
    """Get the status of data in the database."""
    from database.connection import get_session
    from database.models import StockDaily, MarketStatus, StockPool

    with get_session() as session:
        # Count records in stock_daily
        stock_count = session.query(StockDaily).count()

        # Get date range
        latest = session.query(StockDaily.trade_date).order_by(StockDaily.trade_date.desc()).first()
        earliest = session.query(StockDaily.trade_date).order_by(StockDaily.trade_date.asc()).first()

        # Stock pool count
        pool_count = session.query(StockPool).count()

        # Get update status
        status_repo = MarketStatusRepository()
        update_status = None
        try:
            last_update = status_repo.get_last_update("daily_data_update")
            if last_update:
                update_status = {
                    "last_update": last_update.isoformat(),
                    "status": "success"
                }
        except:
            pass

        return {
            "stock_daily_records": stock_count,
            "stock_pool_count": pool_count,
            "date_range": {
                "earliest": str(earliest[0]) if earliest else None,
                "latest": str(latest[0]) if latest else None
            },
            "last_update": update_status
        }


@app.post("/api/admin/init-stock-pool")
def init_stock_pool(background_tasks: BackgroundTasks):
    """
    Initialize the stock pool with CSI 300 stocks.
    """
    def init_task():
        data_service = get_data_service()
        count = data_service.init_stock_pool()
        logger.info(f"Stock pool initialized with {count} stocks")

    background_tasks.add_task(init_task)

    return {
        "status": "started",
        "message": "Stock pool initialization started in background"
    }


# --- USER SUBSCRIPTION (MOCK) ---

@app.post("/api/user/subscribe")
def subscribe(payload: Dict[str, Any]):
    """Handle subscription request."""
    return {
        "status": "success",
        "message": "Subscription processed",
        "subscription": {
            "plan": payload.get("plan", "plus"),
            "strategy_id": payload.get("strategy_id"),
            "expires_at": "2025-02-28"
        }
    }


@app.get("/api/user/subscription-status")
def get_subscription_status():
    """Get user subscription status."""
    return {
        "is_subscribed": False,
        "plan": "free",
        "subscribed_strategies": []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
