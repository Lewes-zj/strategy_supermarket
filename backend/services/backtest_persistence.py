"""回测结果持久化服务"""
from datetime import date
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from database.models import StrategyTrade, StrategyDailySnapshot, StrategyDailyEquity
from engine.models import BacktestResult, Fill


class BacktestPersistenceService:
    """回测结果持久化服务"""

    def __init__(self, db: Session):
        self.db = db

    def save_backtest_result(
        self,
        strategy_id: str,
        result: BacktestResult,
        sector_map: Dict[str, str] = None
    ) -> None:
        """保存回测结果到数据库"""
        sector_map = sector_map or {}

        # 清空该策略的回测数据
        self._clear_backtest_data(strategy_id)

        # 保存交易记录
        self._save_trades(strategy_id, result.trades, sector_map)

        # 保存每日权益和持仓快照
        self._save_daily_data(strategy_id, result, sector_map)

        self.db.commit()

    def _clear_backtest_data(self, strategy_id: str) -> None:
        """清空策略的回测数据"""
        self.db.query(StrategyTrade).filter(
            and_(
                StrategyTrade.strategy_id == strategy_id,
                StrategyTrade.source == "backtest"
            )
        ).delete()
        self.db.query(StrategyDailySnapshot).filter_by(strategy_id=strategy_id).delete()
        self.db.query(StrategyDailyEquity).filter_by(strategy_id=strategy_id).delete()

    def _save_trades(
        self,
        strategy_id: str,
        trades: List,
        sector_map: Dict[str, str]
    ) -> None:
        """保存交易记录

        Handles both Fill objects (from fresh backtest) and dicts (from cache).
        """
        from datetime import datetime

        for fill in trades:
            # Handle both Fill objects and dict format (from cache)
            if isinstance(fill, dict):
                # Cached format: {"timestamp": "...", "symbol": "...", "side": "..."}
                symbol = fill.get("symbol", "")
                timestamp_str = fill.get("timestamp", "")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")) if timestamp_str else None
                except (ValueError, AttributeError):
                    timestamp = None
                trade = StrategyTrade(
                    strategy_id=strategy_id,
                    trade_date=timestamp.date() if timestamp else None,
                    trade_time=timestamp.strftime("%H:%M:%S") if timestamp else None,
                    symbol=symbol,
                    sector=sector_map.get(symbol, "其他"),
                    side=fill.get("side", "buy"),
                    price=0.0,  # Not available in cached format
                    quantity=0,  # Not available in cached format
                    amount=0.0,  # Not available in cached format
                    commission=0.0,  # Not available in cached format
                    source="backtest"
                )
            else:
                # Fill object format (from fresh backtest)
                symbol = fill.order.symbol
                trade = StrategyTrade(
                    strategy_id=strategy_id,
                    trade_date=fill.timestamp.date(),
                    trade_time=fill.timestamp.strftime("%H:%M:%S") if fill.timestamp else None,
                    symbol=symbol,
                    sector=sector_map.get(symbol, "其他"),
                    side=fill.order.side.value,
                    price=float(fill.fill_price),
                    quantity=int(fill.fill_quantity),
                    amount=float(fill.fill_price * fill.fill_quantity),
                    commission=float(fill.commission),
                    source="backtest"
                )
            self.db.add(trade)

    def _save_daily_data(
        self,
        strategy_id: str,
        result: BacktestResult,
        sector_map: Dict[str, str]
    ) -> None:
        """保存每日权益和持仓快照"""
        equity_curve = result.equity_curve
        initial_equity = float(equity_curve["equity"].iloc[0])

        for idx, row in equity_curve.iterrows():
            equity_date = idx.date() if hasattr(idx, 'date') else idx
            total_equity = float(row["equity"])

            # 计算每日收益率
            daily_pnl_pct = float(row.get("returns", 0)) if "returns" in row else 0
            total_pnl_pct = (total_equity - initial_equity) / initial_equity if initial_equity else 0

            equity_record = StrategyDailyEquity(
                strategy_id=strategy_id,
                equity_date=equity_date,
                total_equity=total_equity,
                cash=total_equity * 0.1,  # 简化：假设10%现金
                position_value=total_equity * 0.9,
                daily_pnl_pct=daily_pnl_pct,
                total_pnl_pct=total_pnl_pct,
                position_count=len(result.positions)
            )
            self.db.add(equity_record)
