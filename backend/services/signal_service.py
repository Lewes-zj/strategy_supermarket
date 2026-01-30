"""
Signal Service for Strategy Supermarket.
Manages real-time trading signals and notifications.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database.repository import SignalRepository
from database.models import StrategySignal
from strategies.registry import get_strategy_info, create_strategy

logger = logging.getLogger(__name__)


class SignalService:
    """
    Service for generating and managing trading signals.

    Features:
    - Generate signals from strategies
    - Track active signals
    - Check for signal triggers
    """

    def __init__(self):
        self.signal_repo = SignalRepository()
        self.active_signals: Dict[str, List[StrategySignal]] = {}

    def generate_signals(
        self,
        strategy_id: str,
        current_data: dict,
        context: dict = None
    ) -> List[Dict]:
        """
        Generate trading signals for a strategy.

        Args:
            strategy_id: Strategy identifier
            current_data: Current market data
            context: Additional context (indicators, etc.)

        Returns:
            List of signal dictionaries
        """
        signals = []

        try:
            # Get strategy info
            strategy_info = get_strategy_info(strategy_id)
            if not strategy_info:
                logger.warning(f"Unknown strategy: {strategy_id}")
                return signals

            # Parse signal from data
            signal_type = current_data.get('signal_type')  # 'buy' or 'sell'
            symbol = current_data.get('symbol')
            price = current_data.get('price', 0)
            reason = current_data.get('reason', '')

            if signal_type and symbol:
                signal = {
                    'strategy_id': strategy_id,
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'price': price,
                    'reason': reason,
                    'timestamp': datetime.now()
                }
                signals.append(signal)

                # Save to database
                self.signal_repo.create_signal(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    signal_type=signal_type,
                    price=price,
                    reason=reason
                )

                logger.info(f"Generated {signal_type} signal for {symbol} in {strategy_id}")

        except Exception as e:
            logger.error(f"Error generating signals for {strategy_id}: {e}")

        return signals

    def get_active_signals(self, strategy_id: str) -> List[Dict]:
        """
        Get all active signals for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            List of active signal dictionaries
        """
        try:
            db_signals = self.signal_repo.get_active_signals(strategy_id)

            return [{
                'id': s.id,
                'symbol': s.symbol,
                'signal_type': s.signal_type,
                'price': s.price,
                'quantity': s.quantity,
                'reason': s.reason,
                'created_at': s.created_at.isoformat(),
                'is_active': s.is_active
            } for s in db_signals]

        except Exception as e:
            logger.error(f"Error getting active signals for {strategy_id}: {e}")
            return []

    def close_signal(self, signal_id: int, price: float) -> bool:
        """
        Close an active signal.

        Args:
            signal_id: Signal ID to close
            price: Exit price

        Returns:
            True if successful
        """
        try:
            self.signal_repo.close_signal(signal_id, price)
            logger.info(f"Closed signal {signal_id} at {price}")
            return True

        except Exception as e:
            logger.error(f"Error closing signal {signal_id}: {e}")
            return False

    def check_signal_trigger(
        self,
        strategy_id: str,
        current_price: float,
        position_data: dict
    ) -> Optional[Dict]:
        """
        Check if a signal should be triggered based on current conditions.

        Args:
            strategy_id: Strategy identifier
            current_price: Current market price
            position_data: Current position data

        Returns:
            Signal dict if triggered, None otherwise
        """
        try:
            # Get strategy info to determine trigger logic
            strategy_info = get_strategy_info(strategy_id)
            if not strategy_info:
                return None

            # Example trigger logic (would be strategy-specific)
            stop_loss = position_data.get('stop_loss', 0)
            take_profit = position_data.get('take_profit', 0)
            entry_price = position_data.get('entry_price', 0)

            if entry_price > 0:
                # Calculate PnL percentage
                pnl_pct = (current_price - entry_price) / entry_price

                # Check stop loss
                if pnl_pct <= stop_loss:
                    return {
                        'strategy_id': strategy_id,
                        'signal_type': 'sell',
                        'reason': 'stop_loss',
                        'price': current_price
                    }

                # Check take profit
                if pnl_pct >= take_profit:
                    return {
                        'strategy_id': strategy_id,
                        'signal_type': 'sell',
                        'reason': 'take_profit',
                        'price': current_price
                    }

            return None

        except Exception as e:
            logger.error(f"Error checking signal trigger: {e}")
            return None

    def get_latest_signals(
        self,
        strategy_id: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get latest signals across all strategies or specific strategy.

        Args:
            strategy_id: Filter by strategy (None = all)
            limit: Maximum number of signals to return

        Returns:
            List of signal dictionaries
        """
        # This would query the database for recent signals
        # For now, return from active cache
        signals = []

        if strategy_id:
            active = self.get_active_signals(strategy_id)
            signals.extend(active)
        else:
            # Get signals from all strategies
            from strategies.registry import get_all_strategies
            for strategy_info in get_all_strategies():
                active = self.get_active_signals(strategy_info.strategy_id)
                signals.extend(active)

        # Sort by created_at and limit
        signals.sort(key=lambda x: x['created_at'], reverse=True)
        return signals[:limit]

    def get_signal_summary(self, strategy_id: str) -> Dict:
        """
        Get summary of signals for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Summary dict with statistics
        """
        try:
            active_signals = self.get_active_signals(strategy_id)

            buy_count = sum(1 for s in active_signals if s['signal_type'] == 'buy')
            sell_count = sum(1 for s in active_signals if s['signal_type'] == 'sell')

            return {
                'strategy_id': strategy_id,
                'active_signals': len(active_signals),
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'last_update': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting signal summary for {strategy_id}: {e}")
            return {}

    def is_signal_recent(self, strategy_id: str, minutes: int = 15) -> bool:
        """
        Check if there was a recent signal for a strategy.

        Args:
            strategy_id: Strategy identifier
            minutes: Minutes to look back

        Returns:
            True if signal was generated within time window
        """
        try:
            active_signals = self.get_active_signals(strategy_id)

            if not active_signals:
                return False

            # Check most recent signal
            latest_signal = active_signals[0]
            created_at = datetime.fromisoformat(latest_signal['created_at'])
            time_diff = (datetime.now() - created_at).total_seconds() / 60

            return time_diff <= minutes

        except Exception as e:
            logger.error(f"Error checking signal recency: {e}")
            return False


# Global service instance
_signal_service = None


def get_signal_service() -> SignalService:
    """Get the global signal service instance."""
    global _signal_service

    if _signal_service is None:
        _signal_service = SignalService()

    return _signal_service
