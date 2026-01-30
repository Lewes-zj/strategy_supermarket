from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None

@dataclass
class Fill:
    order: Order
    fill_price: Decimal
    fill_quantity: Decimal
    commission: Decimal
    slippage: Decimal
    timestamp: datetime

@dataclass
class Position:
    symbol: str
    quantity: Decimal = Decimal("0")
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    def update(self, fill: Fill) -> None:
        if fill.order.side == OrderSide.BUY:
            new_quantity = self.quantity + fill.fill_quantity
            if new_quantity != 0:
                self.avg_cost = (
                    (self.quantity * self.avg_cost + fill.fill_quantity * fill.fill_price)
                    / new_quantity
                )
            self.quantity = new_quantity
        else:
            self.realized_pnl += fill.fill_quantity * (fill.fill_price - self.avg_cost)
            self.quantity -= fill.fill_quantity

@dataclass
class Portfolio:
    cash: Decimal
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_capital: Decimal = Decimal("0")

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def process_fill(self, fill: Fill) -> None:
        position = self.get_position(fill.order.symbol)
        position.update(fill)

        if fill.order.side == OrderSide.BUY:
            self.cash -= fill.fill_price * fill.fill_quantity + fill.commission
        else:
            self.cash += fill.fill_price * fill.fill_quantity - fill.commission

    def get_equity(self, prices: Dict[str, Decimal]) -> Decimal:
        equity = self.cash
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in prices:
                equity += position.quantity * prices[symbol]
        return equity

class Strategy(ABC):
    @abstractmethod
    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        pass

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        pass

class ExecutionModel(ABC):
    @abstractmethod
    def execute(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        pass

class SimpleExecutionModel(ExecutionModel):
    def __init__(self, slippage_bps: float = 10, commission_per_share: float = 0.01):
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share

    def execute(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        if order.order_type == OrderType.MARKET:
            base_price = Decimal(str(bar["close"])) # Default to close if open not robust, or use open for next bar execution

            # Apply slippage
            slippage_mult = 1 + (self.slippage_bps / 10000)
            if order.side == OrderSide.BUY:
                fill_price = base_price * Decimal(str(slippage_mult))
            else:
                fill_price = base_price / Decimal(str(slippage_mult))

            commission = order.quantity * Decimal(str(self.commission_per_share))
            slippage = abs(fill_price - base_price) * order.quantity

            return Fill(
                order=order,
                fill_price=fill_price,
                fill_quantity=order.quantity,
                commission=commission,
                slippage=slippage,
                timestamp=bar.name if isinstance(bar.name, datetime) else datetime.now()
            )
        return None

class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        execution_model: ExecutionModel,
        initial_capital: Decimal = Decimal("100000")
    ):
        self.strategy = strategy
        self.execution_model = execution_model
        self.portfolio = Portfolio(cash=initial_capital, initial_capital=initial_capital)
        self.equity_curve: List[tuple] = []
        self.trades: List[Fill] = []

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run backtest on OHLCV data with DatetimeIndex."""
        pending_orders: List[Order] = []
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data.set_index('date', inplace=True)
                data.index = pd.to_datetime(data.index)

        for timestamp, bar in data.iterrows():
            # Execute pending orders at today's prices (assuming Market On Open or Close)
            # For simplicity using Close of current bar as execution price for orders generated in PREVIOUS bar
            # But here we are generating orders in CURRENT bar. 
            # Standard pattern: 
            # 1. Open of Today: Execute orders from Yesterday
            # 2. Close of Today: Calculate new signals -> Order for Tomorrow
            
            # Simplified flow: Orders executed at current bar Close immediately (Slight lookahead if using Close to generate signal)
            # Correct flow: Orders generated at T, executed at T+1 Open.
            
            # Let's check pending orders FIRST (from T-1)
            next_pending = []
            for order in pending_orders:
                fill = self.execution_model.execute(order, bar)
                if fill:
                    self.portfolio.process_fill(fill)
                    self.strategy.on_fill(fill)
                    self.trades.append(fill)
                else:
                    next_pending.append(order)
            pending_orders = next_pending

            # Get current prices for equity calculation
            current_close = Decimal(str(bar["close"]))
            prices = {bar["symbol"] if "symbol" in bar else "default": current_close}
            equity = self.portfolio.get_equity(prices)
            self.equity_curve.append((timestamp, float(equity)))

            # Generate new orders for next bar
            # Pass data up to current timestamp to avoid lookahead
            new_orders = self.strategy.on_bar(timestamp, data.loc[:timestamp])
            pending_orders.extend(new_orders)

        return self._create_results()

    def _create_results(self) -> pd.DataFrame:
        equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        equity_df["returns"] = equity_df["equity"].pct_change()
        return equity_df
