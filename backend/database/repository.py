"""
Data access layer for Strategy Supermarket.
Provides high-level methods for database operations.
"""
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import logging

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from .models import StockDaily, StrategyBacktest, StrategySignal, StockPool, MarketStatus, StrategyPosition
from .connection import get_session

logger = logging.getLogger(__name__)


class StockDataRepository:
    """Repository for stock daily price data."""

    @staticmethod
    def save_stock_data(symbol: str, df: pd.DataFrame) -> int:
        """
        Save stock daily data to database.

        Args:
            symbol: Stock symbol (e.g., "000001")
            df: DataFrame with columns [date, open, high, low, close, volume, amount]

        Returns:
            Total number of rows processed (inserts + updates)
        """
        with get_session() as session:
            insert_count = 0
            update_count = 0
            for _, row in df.iterrows():
                # Convert pandas Timestamp to Python date for proper comparison
                trade_date_value = row['date'].date() if hasattr(row['date'], 'date') else row['date']

                # Check if record exists
                existing = session.query(StockDaily).filter(
                    and_(StockDaily.symbol == symbol, StockDaily.trade_date == trade_date_value)
                ).first()

                if existing:
                    # Update existing record
                    existing.open = row['open']
                    existing.high = row['high']
                    existing.low = row['low']
                    existing.close = row['close']
                    existing.volume = int(row['volume'])
                    existing.amount = row.get('amount')
                    existing.updated_at = datetime.now()
                    update_count += 1
                else:
                    # Insert new record
                    record = StockDaily(
                        symbol=symbol,
                        trade_date=trade_date_value,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=int(row['volume']),
                        amount=row.get('amount')
                    )
                    session.add(record)
                    insert_count += 1

            session.commit()
            total_count = insert_count + update_count
            logger.info(f"Saved {symbol}: {insert_count} new, {update_count} updated, {total_count} total")
            return total_count

    @staticmethod
    def get_stock_data(
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get stock data from database.

        Args:
            symbols: List of stock symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns [symbol, date, open, high, low, close, volume, amount]
        """
        with get_session() as session:
            query = session.query(StockDaily).filter(StockDaily.symbol.in_(symbols))

            if start_date:
                query = query.filter(StockDaily.trade_date >= start_date)
            if end_date:
                query = query.filter(StockDaily.trade_date <= end_date)

            results = query.all()

            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([{
                'symbol': r.symbol,
                'date': r.trade_date,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
                'amount': r.amount
            } for r in results])

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df

    @staticmethod
    def get_latest_date(symbol: str) -> Optional[date]:
        """Get the latest trade date for a symbol."""
        with get_session() as session:
            result = session.query(StockDaily.trade_date).filter(
                StockDaily.symbol == symbol
            ).order_by(desc(StockDaily.trade_date)).first()

            return result[0] if result else None

    @staticmethod
    def get_symbols_needing_update(symbols: List[str], days_behind: int = 1) -> List[str]:
        """
        Get symbols that need data update (data is older than specified days).

        Args:
            symbols: List of symbols to check
            days_behind: Number of days behind to trigger update

        Returns:
            List of symbols needing update
        """
        threshold_date = datetime.now() - timedelta(days=days_behind)
        need_update = []

        with get_session() as session:
            for symbol in symbols:
                latest = session.query(StockDaily.trade_date).filter(
                    StockDaily.symbol == symbol
                ).order_by(desc(StockDaily.trade_date)).first()

                if not latest or latest[0] < threshold_date.date():
                    need_update.append(symbol)

        return need_update


class BacktestRepository:
    """Repository for strategy backtest results."""

    @staticmethod
    def save_backtest_result(
        strategy_id: str,
        symbols: List[str],
        start_date: date,
        end_date: date,
        equity_curve: pd.DataFrame,
        metrics: Dict[str, Any],
        trades: List[Any]
    ) -> None:
        """Save backtest result to cache."""
        with get_session() as session:
            # Convert equity curve to JSON
            equity_json = equity_curve.to_json(orient='records', date_format='iso')

            # Check for existing cache
            existing = session.query(StrategyBacktest).filter(
                and_(
                    StrategyBacktest.strategy_id == strategy_id,
                    StrategyBacktest.symbols == json.dumps(symbols),
                    StrategyBacktest.start_date == start_date,
                    StrategyBacktest.end_date == end_date
                )
            ).first()

            if existing:
                existing.equity_curve = equity_json
                existing.metrics = json.dumps(metrics)
                existing.trades = json.dumps(trades, default=str)
                existing.last_updated = datetime.now()
            else:
                record = StrategyBacktest(
                    strategy_id=strategy_id,
                    symbols=json.dumps(symbols),
                    start_date=start_date,
                    end_date=end_date,
                    equity_curve=equity_json,
                    metrics=json.dumps(metrics),
                    trades=json.dumps(trades, default=str)
                )
                session.add(record)

            session.commit()
            logger.info(f"Saved backtest result for {strategy_id}")

    @staticmethod
    def get_backtest_result(
        strategy_id: str,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Optional[Dict[str, Any]]:
        """Get cached backtest result."""
        with get_session() as session:
            result = session.query(StrategyBacktest).filter(
                and_(
                    StrategyBacktest.strategy_id == strategy_id,
                    StrategyBacktest.symbols == json.dumps(symbols),
                    StrategyBacktest.start_date == start_date,
                    StrategyBacktest.end_date == end_date
                )
            ).first()

            if not result:
                return None

            return {
                'equity_curve': pd.read_json(result.equity_curve, convert_dates=['date']),
                'metrics': json.loads(result.metrics),
                'trades': json.loads(result.trades) if result.trades else [],
                'last_updated': result.last_updated
            }


class SignalRepository:
    """Repository for strategy trading signals."""

    @staticmethod
    def create_signal(
        strategy_id: str,
        symbol: str,
        signal_type: str,
        price: float,
        quantity: int = 100,
        reason: str = ""
    ) -> StrategySignal:
        """Create a new trading signal."""
        with get_session() as session:
            signal = StrategySignal(
                strategy_id=strategy_id,
                symbol=symbol,
                signal_type=signal_type,
                price=price,
                quantity=quantity,
                reason=reason
            )
            session.add(signal)
            session.commit()
            session.refresh(signal)
            logger.info(f"Created {signal_type} signal for {symbol} in {strategy_id}")
            return signal

    @staticmethod
    def get_active_signals(strategy_id: str) -> List[StrategySignal]:
        """Get all active (unclosed) signals for a strategy."""
        with get_session() as session:
            return session.query(StrategySignal).filter(
                and_(StrategySignal.strategy_id == strategy_id, StrategySignal.is_active == True)
            ).order_by(desc(StrategySignal.created_at)).all()

    @staticmethod
    def close_signal(signal_id: int, price: float) -> None:
        """Close a signal (mark as inactive)."""
        with get_session() as session:
            signal = session.query(StrategySignal).filter(
                StrategySignal.id == signal_id
            ).first()

            if signal:
                signal.is_active = False
                signal.closed_at = datetime.now()
                session.commit()
                logger.info(f"Closed signal {signal_id}")


class StockPoolRepository:
    """Repository for stock pool management."""

    @staticmethod
    def add_symbols(symbols: List[Dict[str, str]], index_name: str = "CSI300") -> int:
        """
        Add symbols to stock pool.

        Args:
            symbols: List of dicts with keys: symbol, name, sector
            index_name: Index name (e.g., "CSI300")

        Returns:
            Number of symbols added
        """
        with get_session() as session:
            added_count = 0
            for sym in symbols:
                existing = session.query(StockPool).filter(
                    StockPool.symbol == sym['symbol']
                ).first()

                if not existing:
                    record = StockPool(
                        symbol=sym['symbol'],
                        name=sym.get('name', ''),
                        sector=sym.get('sector', ''),
                        index_name=index_name,
                        is_active=True
                    )
                    session.add(record)
                    added_count += 1
                else:
                    # Update existing
                    existing.name = sym.get('name', existing.name)
                    existing.sector = sym.get('sector', existing.sector)
                    existing.updated_at = datetime.now()

            session.commit()
            logger.info(f"Added/Updated {added_count} symbols to stock pool")
            return added_count

    @staticmethod
    def get_active_symbols(index_name: Optional[str] = None) -> List[str]:
        """Get all active symbols from stock pool."""
        with get_session() as session:
            query = session.query(StockPool.symbol).filter(StockPool.is_active == True)

            if index_name:
                query = query.filter(StockPool.index_name == index_name)

            results = query.all()
            return [r[0] for r in results]

    @staticmethod
    def get_stock_pool() -> List[Dict[str, Any]]:
        """Get all stock pool data."""
        with get_session() as session:
            results = session.query(StockPool).filter(StockPool.is_active == True).all()

            return [{
                'symbol': r.symbol,
                'name': r.name,
                'sector': r.sector,
                'index_name': r.index_name
            } for r in results]


class MarketStatusRepository:
    """Repository for market status tracking."""

    @staticmethod
    def update_status(data_type: str, status: str, error_message: str = None) -> None:
        """Update data processing status."""
        with get_session() as session:
            record = session.query(MarketStatus).filter(
                MarketStatus.data_type == data_type
            ).first()

            now = datetime.now()

            if record:
                record.status = status
                record.error_message = error_message
                if status == "success":
                    record.last_update = now
            else:
                record = MarketStatus(
                    data_type=data_type,
                    status=status,
                    error_message=error_message,
                    last_update=now if status == "success" else None
                )
                session.add(record)

            session.commit()

    @staticmethod
    def get_last_update(data_type: str) -> Optional[datetime]:
        """Get last successful update time for a data type."""
        with get_session() as session:
            result = session.query(MarketStatus.last_update).filter(
                and_(
                    MarketStatus.data_type == data_type,
                    MarketStatus.status == "success"
                )
            ).order_by(desc(MarketStatus.last_update)).first()

            return result[0] if result else None


class PositionRepository:
    """Repository for strategy position tracking."""

    @staticmethod
    def upsert_position(
        strategy_id: str,
        symbol: str,
        sector: str,
        direction: str,
        quantity: int,
        entry_price: float,
        current_price: float,
        days_held: int,
        weight: float,
        floating_pnl: float
    ) -> StrategyPosition:
        """
        Create or update a position for a strategy.

        Args:
            strategy_id: Strategy identifier
            symbol: Stock symbol
            sector: Sector name
            direction: Long/Short
            quantity: Position quantity
            entry_price: Average entry price
            current_price: Current market price
            days_held: Days position has been held
            weight: Position weight percentage
            floating_pnl: Floating P&L percentage

        Returns:
            The created or updated position
        """
        with get_session() as session:
            # Check for existing position
            existing = session.query(StrategyPosition).filter(
                and_(
                    StrategyPosition.strategy_id == strategy_id,
                    StrategyPosition.symbol == symbol
                )
            ).first()

            if existing:
                # Update existing position
                existing.sector = sector
                existing.direction = direction
                existing.quantity = quantity
                existing.entry_price = entry_price
                existing.current_price = current_price
                existing.days_held = days_held
                existing.weight = weight
                existing.floating_pnl = floating_pnl
                existing.updated_at = datetime.now()
                position = existing
            else:
                # Create new position
                position = StrategyPosition(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    sector=sector,
                    direction=direction,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=current_price,
                    days_held=days_held,
                    weight=weight,
                    floating_pnl=floating_pnl
                )
                session.add(position)

            session.commit()
            session.refresh(position)
            return position

    @staticmethod
    def get_positions(strategy_id: str) -> List[StrategyPosition]:
        """Get all positions for a strategy."""
        with get_session() as session:
            return session.query(StrategyPosition).filter(
                StrategyPosition.strategy_id == strategy_id
            ).all()

    @staticmethod
    def delete_position(strategy_id: str, symbol: str) -> bool:
        """Delete a position (when closed)."""
        with get_session() as session:
            position = session.query(StrategyPosition).filter(
                and_(
                    StrategyPosition.strategy_id == strategy_id,
                    StrategyPosition.symbol == symbol
                )
            ).first()

            if position:
                session.delete(position)
                session.commit()
                logger.info(f"Deleted position for {symbol} in {strategy_id}")
                return True
            return False

    @staticmethod
    def clear_strategy_positions(strategy_id: str) -> int:
        """Clear all positions for a strategy."""
        with get_session() as session:
            count = session.query(StrategyPosition).filter(
                StrategyPosition.strategy_id == strategy_id
            ).delete()
            session.commit()
            return count
