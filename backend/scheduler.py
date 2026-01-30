"""
Background task scheduler for Strategy Supermarket.
Handles periodic data updates and signal generation.
"""
import logging
from datetime import datetime, time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import Dict

import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import config
from services.data_service import get_data_service
from database.repository import MarketStatusRepository, StockPoolRepository, SignalRepository, PositionRepository
from database.models import RealtimePrice
from database.connection import get_session
from sqlalchemy import and_

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler = None

# Global cache for realtime prices (accessible by API)
_REALTIME_PRICE_CACHE: Dict[str, Dict] = {}


def get_scheduler() -> BackgroundScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler

    if _scheduler is None:
        _scheduler = BackgroundScheduler(
            timezone='Asia/Shanghai',
            job_defaults={
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 3600
            }
        )

        # Register jobs
        _register_jobs(_scheduler)

        logger.info("Scheduler created and jobs registered")

    return _scheduler


def _register_jobs(scheduler: BackgroundScheduler):
    """Register all scheduled jobs."""

    # Daily stock data update (runs at 15:30 after market close)
    scheduler.add_job(
        _update_daily_data,
        trigger=CronTrigger(hour=15, minute=30),
        id='daily_data_update',
        name='Daily Stock Data Update',
        replace_existing=True
    )

    # Real-time price update (every 5 seconds during trading hours)
    scheduler.add_job(
        _update_realtime_prices,
        trigger='interval',
        seconds=5,
        id='realtime_price_update',
        name='Real-time Price Update',
        replace_existing=True
    )

    # Strategy signal generation (every 5 minutes)
    scheduler.add_job(
        _generate_strategy_signals,
        trigger='interval',
        minutes=5,
        id='signal_generation',
        name='Strategy Signal Generation',
        replace_existing=True
    )

    # Stock pool sync (weekly on Sunday at 2 AM)
    scheduler.add_job(
        _sync_stock_pool,
        trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
        id='stock_pool_sync',
        name='Stock Pool Sync',
        replace_existing=True
    )

    logger.info(f"Registered {len(scheduler.get_jobs())} scheduled jobs")


def _update_daily_data():
    """Update daily stock data for all symbols."""
    try:
        logger.info("Starting daily data update...")

        data_service = get_data_service()
        status_repo = MarketStatusRepository()

        status_repo.update_status("daily_data_update", "running")

        # Update data for all active symbols (10 years historical)
        stats = data_service.update_stock_data(days_back=3650)

        status_repo.update_status("daily_data_update", "success")

        logger.info(f"Daily data update completed: {stats}")

    except Exception as e:
        logger.error(f"Error in daily data update: {e}")
        status_repo = MarketStatusRepository()
        status_repo.update_status("daily_data_update", "failed", str(e))


def _update_realtime_prices():
    """Update real-time prices for all stock pool symbols (only during trading hours)."""
    try:
        # Check if market is open
        now = datetime.now()
        current_time = now.time()

        # Trading hours: 9:30-11:30, 13:00-15:00
        morning_start = time(9, 30)
        morning_end = time(11, 30)
        afternoon_start = time(13, 0)
        afternoon_end = time(15, 0)

        is_trading_time = (
            (morning_start <= current_time <= morning_end) or
            (afternoon_start <= current_time <= afternoon_end)
        ) and now.weekday() < 5  # Weekdays only

        if not is_trading_time:
            return

        # Get all symbols from stock pool
        stock_pool_repo = StockPoolRepository()
        symbols_data = stock_pool_repo.get_stock_pool()
        symbols = [s['symbol'] for s in symbols_data]

        if not symbols:
            return

        # Fetch realtime prices using AkShare spot data
        import akshare as ak
        import pandas as pd

        try:
            # Fetch all A-share spot data
            spot_df = ak.stock_zh_a_spot_em()

            if spot_df is None or spot_df.empty:
                logger.debug("No spot data returned")
                return

            # Process each symbol in our stock pool
            updated_count = 0
            for symbol in symbols:
                # Find the symbol in spot data (convert to 6-digit format with prefix)
                # Spot data uses format like '000001' for 深市, '600000' for 沪市
                match = spot_df[spot_df['代码'] == symbol]

                if not match.empty:
                    row = match.iloc[0]
                    price_data = {
                        'symbol': symbol,
                        'price': float(row['最新价']),
                        'change': float(row['涨跌额']),
                        'change_pct': float(row['涨跌幅']),
                        'volume': float(row['成交量']),
                        'amount': float(row['成交额']),
                        'high': float(row['最高']),
                        'low': float(row['最低']),
                        'open': float(row['今开']),
                        'prev_close': float(row['昨收']),
                        'timestamp': now
                    }

                    # Update global cache
                    _REALTIME_PRICE_CACHE[symbol] = price_data

                    # Save to database
                    _save_realtime_price(symbol, price_data)
                    updated_count += 1

            if updated_count > 0:
                logger.debug(f"Updated {updated_count} realtime prices")

        except Exception as e:
            logger.error(f"Error fetching spot data: {e}")

    except Exception as e:
        logger.error(f"Error in real-time price update: {e}")


def _save_realtime_price(symbol: str, price_data: Dict):
    """Save realtime price to database."""
    try:
        with get_session() as session:
            # Check if exists
            existing = session.query(RealtimePrice).filter(
                RealtimePrice.symbol == symbol
            ).first()

            if existing:
                existing.price = price_data['price']
                existing.change = price_data['change']
                existing.change_pct = price_data['change_pct']
                existing.volume = price_data['volume']
                existing.amount = price_data['amount']
                existing.high = price_data['high']
                existing.low = price_data['low']
                existing.open = price_data['open']
                existing.prev_close = price_data['prev_close']
                existing.updated_at = datetime.now()
            else:
                record = RealtimePrice(
                    symbol=symbol,
                    price=price_data['price'],
                    change=price_data['change'],
                    change_pct=price_data['change_pct'],
                    volume=price_data['volume'],
                    amount=price_data['amount'],
                    high=price_data['high'],
                    low=price_data['low'],
                    open=price_data['open'],
                    prev_close=price_data['prev_close']
                )
                session.add(record)

            session.commit()
    except Exception as e:
        logger.error(f"Error saving realtime price for {symbol}: {e}")


def get_realtime_prices() -> Dict[str, Dict]:
    """Get all cached realtime prices."""
    return _REALTIME_PRICE_CACHE.copy()


def get_realtime_price(symbol: str) -> Dict:
    """Get realtime price for a specific symbol."""
    return _REALTIME_PRICE_CACHE.get(symbol)


def _generate_strategy_signals():
    """Generate trading signals for all active strategies."""
    try:
        logger.info("Generating strategy signals...")

        # Get all active strategies
        from strategies.registry import get_all_strategies, create_strategy, get_strategy_info
        from engine.backtester import Backtester, SimpleExecutionModel
        from engine.data_loader import generate_mock_data
        from decimal import Decimal
        import pandas as pd

        signal_repo = SignalRepository()
        position_repo = PositionRepository()
        stock_pool_repo = StockPoolRepository()

        # Get stock pool for sector mapping
        stock_pool = stock_pool_repo.get_stock_pool()
        sector_map = {s['symbol']: s.get('sector', '其他') for s in stock_pool}

        for strategy_info in get_all_strategies():
            strategy_id = strategy_info.strategy_id
            symbols = strategy_info.default_symbols[:1]  # Use first symbol for signal generation

            try:
                # Create strategy instance
                strategy = create_strategy(strategy_id)

                # Create execution model and backtester
                execution_model = SimpleExecutionModel()
                backtester = Backtester(
                    strategy=strategy,
                    execution_model=execution_model,
                    initial_capital=Decimal("100000")
                )

                # Generate mock data for the strategy
                symbol = symbols[0] if symbols else "DEMO"
                data = generate_mock_data(start_date="20230101", days=730, symbol=symbol)

                # Run backtest
                backtester.run(data)

                # Clear old positions for this strategy
                position_repo.clear_strategy_positions(strategy_id)

                # Get current positions from backtest
                for pos_symbol, position in backtester.portfolio.positions.items():
                    if position.quantity > 0:
                        entry_price = float(position.avg_cost) if position.avg_cost else 0

                        # Calculate current price (use entry price for demo)
                        current_price = entry_price * (1 + 0.02 * (hash(pos_symbol) % 10 - 5))

                        # Calculate P&L
                        if entry_price > 0:
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = 0.0

                        # Calculate position weight
                        # Get final equity from equity curve
                        if backtester.equity_curve:
                            total_value = float(backtester.equity_curve[-1][1])  # (timestamp, equity)
                        else:
                            total_value = 100000.0  # Initial capital
                        position_value = float(position.quantity) * current_price
                        weight = (position_value / total_value * 100) if total_value > 0 else 0

                        # Update or create position in database
                        position_repo.upsert_position(
                            strategy_id=strategy_id,
                            symbol=pos_symbol,
                            sector=sector_map.get(pos_symbol, '其他'),
                            direction="Long",
                            quantity=int(position.quantity),
                            entry_price=entry_price,
                            current_price=current_price,
                            days_held=1,  # New position
                            weight=round(weight, 2),
                            floating_pnl=round(pnl_pct, 2)
                        )

                        logger.debug(f"Updated position for {pos_symbol} in {strategy_id}: {position.quantity} shares")

                # Generate signals from recent trades
                recent_trades = backtester.trades[-5:] if backtester.trades else []

                for trade in recent_trades:
                    # Only create signals for recent trades (within last hour)
                    trade_age = (datetime.now() - trade.timestamp).total_seconds() / 60
                    if trade_age <= 60:  # Within last hour
                        signal_repo.create_signal(
                            strategy_id=strategy_id,
                            symbol=trade.order.symbol,
                            signal_type=trade.order.side.value,
                            price=float(trade.fill_price),
                            quantity=int(trade.fill_quantity),
                            reason=f"{strategy_info.name} signal"
                        )
                        logger.info(f"Created {trade.order.side.value} signal for {trade.order.symbol} in {strategy_id}")

            except Exception as e:
                logger.error(f"Error generating signals for {strategy_id}: {e}")
                import traceback
                traceback.print_exc()

        logger.debug("Strategy signal generation completed")

    except Exception as e:
        logger.error(f"Error in signal generation: {e}")
        import traceback
        traceback.print_exc()


def _sync_stock_pool():
    """Synchronize stock pool with latest CSI 300 constituents."""
    try:
        logger.info("Syncing stock pool...")

        data_service = get_data_service()
        status_repo = MarketStatusRepository()

        status_repo.update_status("stock_pool_sync", "running")

        # Re-initialize stock pool
        count = data_service.init_stock_pool()

        status_repo.update_status("stock_pool_sync", "success")

        logger.info(f"Stock pool sync completed: {count} symbols")

    except Exception as e:
        logger.error(f"Error in stock pool sync: {e}")
        status_repo = MarketStatusRepository()
        status_repo.update_status("stock_pool_sync", "failed", str(e))


def start_scheduler():
    """Start the background scheduler."""
    if not config.SCHEDULER_ENABLED:
        logger.info("Scheduler is disabled in config")
        return

    try:
        scheduler = get_scheduler()
        scheduler.start()

        logger.info("Scheduler started successfully")

        # Log next run times
        for job in scheduler.get_jobs():
            logger.info(f"Job '{job.name}': next run at {job.next_run_time}")

    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")


def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler

    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        logger.info("Scheduler stopped")


def get_job_status() -> dict:
    """Get status of all scheduled jobs."""
    scheduler = get_scheduler()

    jobs_status = {}

    for job in scheduler.get_jobs():
        jobs_status[job.id] = {
            'name': job.name,
            'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
            'running': scheduler.running
        }

    return jobs_status


def trigger_job(job_id: str) -> bool:
    """Manually trigger a scheduled job."""
    try:
        scheduler = get_scheduler()
        job = scheduler.get_job(job_id)

        if job:
            job.modify(next_run_time=datetime.now())
            logger.info(f"Job '{job_id}' triggered manually")
            return True

        return False

    except Exception as e:
        logger.error(f"Error triggering job {job_id}: {e}")
        return False
