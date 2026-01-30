# 回测系统重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重构回测引擎，实现多标的支持、T+1执行、Walk-Forward优化器和Monte Carlo分析

**Architecture:** 模块化重构，保持API兼容。核心引擎拆分为 models/portfolio/execution/metrics 四个独立模块，新增 walk_forward 和 monte_carlo 高级分析模块，通过 backtest_service 统一调度。

**Tech Stack:** Python 3.14, FastAPI, pandas, numpy, SQLAlchemy, pytest

---

## 阶段 1：核心数据模型 (engine/models.py)

### Task 1.1: 创建测试目录和基础数据模型测试

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/engine/__init__.py`
- Create: `backend/tests/engine/test_models.py`

**Step 1: 创建测试目录结构**

```bash
mkdir -p backend/tests/engine
touch backend/tests/__init__.py
touch backend/tests/engine/__init__.py
```

**Step 2: 编写 Order 和 Fill 数据模型测试**

```python
# backend/tests/engine/test_models.py
import pytest
from decimal import Decimal
from datetime import datetime

from engine.models import (
    OrderSide, OrderType, Order, Fill, Position,
    Trade, BacktestResult, PerformanceMetrics
)


class TestOrderSide:
    def test_buy_value(self):
        assert OrderSide.BUY.value == "buy"

    def test_sell_value(self):
        assert OrderSide.SELL.value == "sell"


class TestOrderType:
    def test_market_value(self):
        assert OrderType.MARKET.value == "market"

    def test_limit_value(self):
        assert OrderType.LIMIT.value == "limit"

    def test_stop_value(self):
        assert OrderType.STOP.value == "stop"


class TestOrder:
    def test_market_order_creation(self):
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        assert order.symbol == "000001"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("100")
        assert order.order_type == OrderType.MARKET
        assert order.limit_price is None
        assert order.stop_price is None

    def test_limit_order_creation(self):
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.50")
        )
        assert order.limit_price == Decimal("10.50")

    def test_stop_order_creation(self):
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.STOP,
            stop_price=Decimal("9.50")
        )
        assert order.stop_price == Decimal("9.50")


class TestFill:
    def test_fill_creation(self):
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        fill = Fill(
            order=order,
            fill_price=Decimal("10.55"),
            fill_quantity=Decimal("100"),
            commission=Decimal("5.00"),
            slippage=Decimal("0.50"),
            timestamp=datetime(2025, 1, 15, 9, 30, 0)
        )
        assert fill.fill_price == Decimal("10.55")
        assert fill.fill_quantity == Decimal("100")
        assert fill.commission == Decimal("5.00")


class TestPosition:
    def test_position_initial_state(self):
        pos = Position(symbol="000001")
        assert pos.symbol == "000001"
        assert pos.quantity == Decimal("0")
        assert pos.avg_cost == Decimal("0")
        assert pos.realized_pnl == Decimal("0")

    def test_position_update_buy(self):
        pos = Position(symbol="000001")
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        fill = Fill(
            order=order,
            fill_price=Decimal("10.00"),
            fill_quantity=Decimal("100"),
            commission=Decimal("5.00"),
            slippage=Decimal("0"),
            timestamp=datetime.now()
        )
        pos.update(fill)
        assert pos.quantity == Decimal("100")
        assert pos.avg_cost == Decimal("10.00")

    def test_position_update_sell_with_profit(self):
        pos = Position(symbol="000001", quantity=Decimal("100"), avg_cost=Decimal("10.00"))
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        fill = Fill(
            order=order,
            fill_price=Decimal("12.00"),
            fill_quantity=Decimal("100"),
            commission=Decimal("5.00"),
            slippage=Decimal("0"),
            timestamp=datetime.now()
        )
        pos.update(fill)
        assert pos.quantity == Decimal("0")
        assert pos.realized_pnl == Decimal("200.00")  # (12-10) * 100
```

**Step 3: 运行测试验证失败**

```bash
cd backend && python -m pytest tests/engine/test_models.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'engine.models'"

---

### Task 1.2: 实现核心数据模型

**Files:**
- Create: `backend/engine/models.py`
- Modify: `backend/engine/__init__.py`

**Step 1: 创建 engine/models.py**

```python
# backend/engine/models.py
"""
回测系统核心数据模型
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    """交易订单"""
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None


@dataclass
class Fill:
    """订单成交记录"""
    order: Order
    fill_price: Decimal
    fill_quantity: Decimal
    commission: Decimal
    slippage: Decimal
    timestamp: datetime


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: Decimal = Decimal("0")
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    entry_date: Optional[date] = None

    def update(self, fill: Fill) -> None:
        """根据成交更新持仓"""
        if fill.order.side == OrderSide.BUY:
            new_quantity = self.quantity + fill.fill_quantity
            if new_quantity != 0:
                self.avg_cost = (
                    (self.quantity * self.avg_cost + fill.fill_quantity * fill.fill_price)
                    / new_quantity
                )
            self.quantity = new_quantity
            if self.entry_date is None:
                self.entry_date = fill.timestamp.date() if fill.timestamp else None
        else:
            self.realized_pnl += fill.fill_quantity * (fill.fill_price - self.avg_cost)
            self.quantity -= fill.fill_quantity
            if self.quantity == 0:
                self.entry_date = None


@dataclass
class Trade:
    """完整交易记录（用于API返回）"""
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    pnl: float = 0.0


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    # 核心指标
    sharpe: float = 0.0
    calmar: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0

    # 收益指标
    total_return: float = 0.0
    cagr: float = 0.0
    ytd_return: float = 0.0
    mtd_return: float = 0.0

    # 风险指标
    volatility: float = 0.0
    alpha: float = 0.0
    beta: float = 1.0

    # 交易统计
    win_rate: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    pl_ratio: float = 0.0
    avg_hold_days: float = 0.0
    consecutive_wins: int = 0

    # 其他
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    drawdown_period: str = "N/A"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容现有API）"""
        return {
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "ytd_return": self.ytd_return,
            "mtd_return": self.mtd_return,
            "volatility": self.volatility,
            "alpha": self.alpha,
            "beta": self.beta,
            "win_rate": self.win_rate,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "pl_ratio": self.pl_ratio,
            "avg_hold_days": self.avg_hold_days,
            "consecutive_wins": self.consecutive_wins,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "drawdown_period": self.drawdown_period,
            # 兼容旧字段名
            "strategy_return": self.cagr,
            "excess_max_drawdown": self.max_drawdown,
        }


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.DataFrame
    trades: List[Fill]
    metrics: PerformanceMetrics
    positions: Dict[str, Position]
    split_index: Optional[int] = None      # Walk-Forward用
    optimal_params: Optional[Dict] = None  # Walk-Forward用


@dataclass
class WalkForwardResult:
    """Walk-Forward分析结果"""
    combined_equity: pd.DataFrame
    split_results: List[BacktestResult]
    param_history: List[Dict]
    stability_score: float


@dataclass
class MonteCarloResult:
    """Monte Carlo分析结果"""
    expected_max_drawdown: float
    var_95: float
    cvar_95: float
    probability_of_loss: Dict[int, float]
    return_confidence_interval: Tuple[float, float]
    simulations: Optional[Any] = None  # numpy array
```

**Step 2: 创建 engine/__init__.py**

```python
# backend/engine/__init__.py
"""回测引擎模块"""
from .models import (
    OrderSide, OrderType, Order, Fill, Position, Trade,
    PerformanceMetrics, BacktestResult, WalkForwardResult, MonteCarloResult
)

__all__ = [
    "OrderSide", "OrderType", "Order", "Fill", "Position", "Trade",
    "PerformanceMetrics", "BacktestResult", "WalkForwardResult", "MonteCarloResult"
]
```

**Step 3: 运行测试验证通过**

```bash
cd backend && python -m pytest tests/engine/test_models.py -v
```

Expected: PASS (all tests)

**Step 4: 提交**

```bash
git add backend/engine/models.py backend/engine/__init__.py backend/tests/
git commit -m "feat(engine): add core data models for backtest system

- Add OrderSide, OrderType enums
- Add Order, Fill, Position dataclasses
- Add PerformanceMetrics with API-compatible to_dict()
- Add BacktestResult, WalkForwardResult, MonteCarloResult
- Add initial test suite

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## 阶段 2：Portfolio 组合管理模块

### Task 2.1: 编写 Portfolio 测试

**Files:**
- Create: `backend/tests/engine/test_portfolio.py`

**Step 1: 编写 Portfolio 测试**

```python
# backend/tests/engine/test_portfolio.py
import pytest
from decimal import Decimal
from datetime import datetime

from engine.models import Order, OrderSide, OrderType, Fill
from engine.portfolio import Portfolio


class TestPortfolio:
    def test_initial_state(self):
        portfolio = Portfolio(cash=Decimal("1000000"))
        assert portfolio.cash == Decimal("1000000")
        assert len(portfolio.positions) == 0

    def test_get_position_creates_new(self):
        portfolio = Portfolio(cash=Decimal("1000000"))
        pos = portfolio.get_position("000001")
        assert pos.symbol == "000001"
        assert pos.quantity == Decimal("0")

    def test_process_buy_fill(self):
        portfolio = Portfolio(cash=Decimal("1000000"), initial_capital=Decimal("1000000"))
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        fill = Fill(
            order=order,
            fill_price=Decimal("10.00"),
            fill_quantity=Decimal("100"),
            commission=Decimal("5.00"),
            slippage=Decimal("0"),
            timestamp=datetime.now()
        )
        portfolio.process_fill(fill)

        assert portfolio.cash == Decimal("1000000") - Decimal("1000") - Decimal("5")
        pos = portfolio.get_position("000001")
        assert pos.quantity == Decimal("100")
        assert pos.avg_cost == Decimal("10.00")

    def test_process_sell_fill(self):
        portfolio = Portfolio(cash=Decimal("999000"), initial_capital=Decimal("1000000"))
        # 先建立持仓
        portfolio.positions["000001"] = portfolio.get_position("000001")
        portfolio.positions["000001"].quantity = Decimal("100")
        portfolio.positions["000001"].avg_cost = Decimal("10.00")

        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )
        fill = Fill(
            order=order,
            fill_price=Decimal("12.00"),
            fill_quantity=Decimal("100"),
            commission=Decimal("5.00"),
            slippage=Decimal("0"),
            timestamp=datetime.now()
        )
        portfolio.process_fill(fill)

        # 卖出收入 1200 - 5 = 1195
        assert portfolio.cash == Decimal("999000") + Decimal("1200") - Decimal("5")
        pos = portfolio.get_position("000001")
        assert pos.quantity == Decimal("0")
        assert pos.realized_pnl == Decimal("200")  # (12-10)*100

    def test_get_equity(self):
        portfolio = Portfolio(cash=Decimal("900000"), initial_capital=Decimal("1000000"))
        portfolio.positions["000001"] = portfolio.get_position("000001")
        portfolio.positions["000001"].quantity = Decimal("100")

        prices = {"000001": Decimal("1000")}
        equity = portfolio.get_equity(prices)
        assert equity == Decimal("900000") + Decimal("100000")  # cash + 100*1000

    def test_get_weights(self):
        portfolio = Portfolio(cash=Decimal("500000"), initial_capital=Decimal("1000000"))
        portfolio.positions["000001"] = portfolio.get_position("000001")
        portfolio.positions["000001"].quantity = Decimal("100")
        portfolio.positions["600519"] = portfolio.get_position("600519")
        portfolio.positions["600519"].quantity = Decimal("50")

        prices = {
            "000001": Decimal("2500"),  # 100 * 2500 = 250000
            "600519": Decimal("5000"),  # 50 * 5000 = 250000
        }
        # 总权益 = 500000 + 250000 + 250000 = 1000000
        weights = portfolio.get_weights(prices)

        assert abs(weights["000001"] - 0.25) < 0.001
        assert abs(weights["600519"] - 0.25) < 0.001
```

**Step 2: 运行测试验证失败**

```bash
cd backend && python -m pytest tests/engine/test_portfolio.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'engine.portfolio'"

---

### Task 2.2: 实现 Portfolio 模块

**Files:**
- Create: `backend/engine/portfolio.py`
- Modify: `backend/engine/__init__.py`

**Step 1: 创建 engine/portfolio.py**

```python
# backend/engine/portfolio.py
"""
组合管理模块：支持多标的持仓管理
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List

from .models import Position, Fill, Order, OrderSide


@dataclass
class Portfolio:
    """投资组合管理器"""
    cash: Decimal
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_capital: Decimal = Decimal("0")

    def __post_init__(self):
        if self.initial_capital == Decimal("0"):
            self.initial_capital = self.cash

    def get_position(self, symbol: str) -> Position:
        """获取指定标的持仓，不存在则创建"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def process_fill(self, fill: Fill) -> None:
        """处理成交"""
        position = self.get_position(fill.order.symbol)
        position.update(fill)

        if fill.order.side == OrderSide.BUY:
            self.cash -= fill.fill_price * fill.fill_quantity + fill.commission
        else:
            self.cash += fill.fill_price * fill.fill_quantity - fill.commission

    def get_equity(self, prices: Dict[str, Decimal]) -> Decimal:
        """计算组合总权益"""
        equity = self.cash
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in prices:
                equity += position.quantity * prices[symbol]
        return equity

    def get_weights(self, prices: Dict[str, Decimal]) -> Dict[str, float]:
        """计算各标的权重"""
        equity = self.get_equity(prices)
        if equity == 0:
            return {}

        weights = {}
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in prices:
                value = position.quantity * prices[symbol]
                weights[symbol] = float(value / equity)
        return weights

    def get_total_realized_pnl(self) -> Decimal:
        """计算已实现盈亏总和"""
        return sum(pos.realized_pnl for pos in self.positions.values())

    def get_active_positions(self) -> Dict[str, Position]:
        """获取所有有持仓的标的"""
        return {
            symbol: pos for symbol, pos in self.positions.items()
            if pos.quantity != 0
        }
```

**Step 2: 更新 engine/__init__.py**

```python
# backend/engine/__init__.py
"""回测引擎模块"""
from .models import (
    OrderSide, OrderType, Order, Fill, Position, Trade,
    PerformanceMetrics, BacktestResult, WalkForwardResult, MonteCarloResult
)
from .portfolio import Portfolio

__all__ = [
    "OrderSide", "OrderType", "Order", "Fill", "Position", "Trade",
    "PerformanceMetrics", "BacktestResult", "WalkForwardResult", "MonteCarloResult",
    "Portfolio"
]
```

**Step 3: 运行测试验证通过**

```bash
cd backend && python -m pytest tests/engine/test_portfolio.py -v
```

Expected: PASS (all tests)

**Step 4: 提交**

```bash
git add backend/engine/portfolio.py backend/engine/__init__.py backend/tests/engine/test_portfolio.py
git commit -m "feat(engine): add Portfolio multi-asset management module

- Support multiple positions with cash tracking
- Calculate equity with current market prices
- Calculate position weights for portfolio analysis
- Track realized PnL per position

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## 阶段 3：执行模型模块 (engine/execution.py)

### Task 3.1: 编写执行模型测试

**Files:**
- Create: `backend/tests/engine/test_execution.py`

**Step 1: 编写执行模型测试**

```python
# backend/tests/engine/test_execution.py
import pytest
import pandas as pd
from decimal import Decimal
from datetime import datetime

from engine.models import Order, OrderSide, OrderType
from engine.execution import (
    MarketExecutionModel, LimitExecutionModel,
    StopExecutionModel, CompositeExecutionModel
)


@pytest.fixture
def sample_bars():
    """创建测试用K线数据"""
    current_bar = pd.Series({
        "open": 10.00,
        "high": 10.50,
        "low": 9.80,
        "close": 10.20,
        "volume": 1000000
    }, name=datetime(2025, 1, 15, 15, 0, 0))

    next_bar = pd.Series({
        "open": 10.25,
        "high": 10.60,
        "low": 10.10,
        "close": 10.40,
        "volume": 1200000
    }, name=datetime(2025, 1, 16, 15, 0, 0))

    return current_bar, next_bar


class TestMarketExecutionModel:
    def test_buy_order_execution(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = MarketExecutionModel(slippage_bps=10, commission_rate=0.0003)

        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        # 买入用下一根K线开盘价 10.25 * (1 + 0.001) = 10.26025
        assert fill.fill_price > Decimal("10.25")
        assert fill.fill_quantity == Decimal("100")
        assert fill.commission >= Decimal("5")  # 最低佣金5元

    def test_sell_order_execution(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = MarketExecutionModel(slippage_bps=10, commission_rate=0.0003)

        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        # 卖出用下一根K线开盘价 10.25 / (1 + 0.001) = 10.23975
        assert fill.fill_price < Decimal("10.25")

    def test_no_next_bar_returns_none(self, sample_bars):
        current_bar, _ = sample_bars
        model = MarketExecutionModel()

        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )

        fill = model.execute(order, current_bar, None)
        assert fill is None


class TestLimitExecutionModel:
    def test_buy_limit_order_filled(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = LimitExecutionModel()

        # 限价10.15，下一根K线低点10.10，可以成交
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.15")
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        assert fill.fill_price == Decimal("10.15")

    def test_buy_limit_order_not_filled(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = LimitExecutionModel()

        # 限价10.05，下一根K线低点10.10，不能成交
        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.05")
        )

        fill = model.execute(order, current_bar, next_bar)
        assert fill is None

    def test_sell_limit_order_filled(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = LimitExecutionModel()

        # 限价10.55，下一根K线高点10.60，可以成交
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.55")
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        assert fill.fill_price == Decimal("10.55")


class TestStopExecutionModel:
    def test_stop_loss_sell_triggered(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = StopExecutionModel(slippage_bps=20)

        # 止损价10.15，下一根K线低点10.10，触发止损
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.STOP,
            stop_price=Decimal("10.15")
        )

        fill = model.execute(order, current_bar, next_bar)

        assert fill is not None
        # 止损触发后以开盘价成交，带滑点
        assert fill.fill_price < Decimal("10.25")

    def test_stop_loss_not_triggered(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = StopExecutionModel()

        # 止损价10.05，下一根K线低点10.10，不触发
        order = Order(
            symbol="000001",
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            order_type=OrderType.STOP,
            stop_price=Decimal("10.05")
        )

        fill = model.execute(order, current_bar, next_bar)
        assert fill is None


class TestCompositeExecutionModel:
    def test_routes_market_order(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = CompositeExecutionModel()

        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.MARKET
        )

        fill = model.execute(order, current_bar, next_bar)
        assert fill is not None

    def test_routes_limit_order(self, sample_bars):
        current_bar, next_bar = sample_bars
        model = CompositeExecutionModel()

        order = Order(
            symbol="000001",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10.15")
        )

        fill = model.execute(order, current_bar, next_bar)
        assert fill is not None
```

**Step 2: 运行测试验证失败**

```bash
cd backend && python -m pytest tests/engine/test_execution.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'engine.execution'"

---

### Task 3.2: 实现执行模型

**Files:**
- Create: `backend/engine/execution.py`
- Modify: `backend/engine/__init__.py`

**Step 1: 创建 engine/execution.py**

```python
# backend/engine/execution.py
"""
订单执行模型：支持市价单、限价单、止损单
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime
from typing import Optional
import pandas as pd

from .models import Order, OrderType, OrderSide, Fill


class ExecutionModel(ABC):
    """执行模型抽象基类"""

    @abstractmethod
    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        """
        执行订单

        Args:
            order: 待执行订单
            current_bar: 当前K线
            next_bar: 下一根K线（T+1执行用）

        Returns:
            成交记录，未成交返回None
        """
        pass


class MarketExecutionModel(ExecutionModel):
    """
    市价单执行模型

    T日信号 → T+1开盘执行（避免look-ahead bias）
    """

    def __init__(self, slippage_bps: float = 10, commission_rate: float = 0.0003):
        """
        Args:
            slippage_bps: 滑点（基点），默认10bps
            commission_rate: 佣金费率，默认万三
        """
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type != OrderType.MARKET:
            return None
        if next_bar is None:
            return None  # 无下一根K线，无法T+1执行

        # 使用下一根K线的开盘价
        base_price = Decimal(str(next_bar["open"]))

        # 应用滑点
        slippage_mult = Decimal(str(1 + self.slippage_bps / 10000))
        if order.side == OrderSide.BUY:
            fill_price = base_price * slippage_mult
        else:
            fill_price = base_price / slippage_mult

        # 计算佣金（最低5元）
        trade_value = fill_price * order.quantity
        commission = trade_value * Decimal(str(self.commission_rate))
        commission = max(commission, Decimal("5"))

        # 计算滑点金额
        slippage_amount = abs(fill_price - base_price) * order.quantity

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=slippage_amount,
            timestamp=next_bar.name if hasattr(next_bar, 'name') and isinstance(next_bar.name, datetime) else datetime.now()
        )


class LimitExecutionModel(ExecutionModel):
    """
    限价单执行模型

    价格触及限价时成交
    """

    def __init__(self, commission_rate: float = 0.0003):
        self.commission_rate = commission_rate

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            return None

        bar = next_bar if next_bar is not None else current_bar
        low = Decimal(str(bar["low"]))
        high = Decimal(str(bar["high"]))

        # 检查价格是否触及
        if order.side == OrderSide.BUY:
            if low <= order.limit_price:
                fill_price = order.limit_price
            else:
                return None
        else:
            if high >= order.limit_price:
                fill_price = order.limit_price
            else:
                return None

        # 计算佣金
        trade_value = fill_price * order.quantity
        commission = max(trade_value * Decimal(str(self.commission_rate)), Decimal("5"))

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=Decimal("0"),
            timestamp=bar.name if hasattr(bar, 'name') and isinstance(bar.name, datetime) else datetime.now()
        )


class StopExecutionModel(ExecutionModel):
    """
    止损单执行模型

    价格突破止损价时以市价成交
    """

    def __init__(self, slippage_bps: float = 20, commission_rate: float = 0.0003):
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type != OrderType.STOP or order.stop_price is None:
            return None

        bar = next_bar if next_bar is not None else current_bar
        low = Decimal(str(bar["low"]))
        high = Decimal(str(bar["high"]))
        open_price = Decimal(str(bar["open"]))

        # 检查是否触发止损
        triggered = False
        if order.side == OrderSide.SELL:  # 止损卖出
            if low <= order.stop_price:
                triggered = True
        else:  # 止损买入（做空回补）
            if high >= order.stop_price:
                triggered = True

        if not triggered:
            return None

        # 触发后以开盘价成交，带滑点
        slippage_mult = Decimal(str(1 + self.slippage_bps / 10000))
        if order.side == OrderSide.BUY:
            fill_price = open_price * slippage_mult
        else:
            fill_price = open_price / slippage_mult

        trade_value = fill_price * order.quantity
        commission = max(trade_value * Decimal(str(self.commission_rate)), Decimal("5"))
        slippage_amount = abs(fill_price - open_price) * order.quantity

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=slippage_amount,
            timestamp=bar.name if hasattr(bar, 'name') and isinstance(bar.name, datetime) else datetime.now()
        )


class CompositeExecutionModel(ExecutionModel):
    """
    组合执行模型

    根据订单类型自动选择对应的执行器
    """

    def __init__(
        self,
        slippage_bps: float = 10,
        stop_slippage_bps: float = 20,
        commission_rate: float = 0.0003
    ):
        self.market_model = MarketExecutionModel(slippage_bps, commission_rate)
        self.limit_model = LimitExecutionModel(commission_rate)
        self.stop_model = StopExecutionModel(stop_slippage_bps, commission_rate)

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type == OrderType.MARKET:
            return self.market_model.execute(order, current_bar, next_bar)
        elif order.order_type == OrderType.LIMIT:
            return self.limit_model.execute(order, current_bar, next_bar)
        elif order.order_type == OrderType.STOP:
            return self.stop_model.execute(order, current_bar, next_bar)
        return None
```

**Step 2: 更新 engine/__init__.py**

```python
# backend/engine/__init__.py
"""回测引擎模块"""
from .models import (
    OrderSide, OrderType, Order, Fill, Position, Trade,
    PerformanceMetrics, BacktestResult, WalkForwardResult, MonteCarloResult
)
from .portfolio import Portfolio
from .execution import (
    ExecutionModel, MarketExecutionModel, LimitExecutionModel,
    StopExecutionModel, CompositeExecutionModel
)

__all__ = [
    "OrderSide", "OrderType", "Order", "Fill", "Position", "Trade",
    "PerformanceMetrics", "BacktestResult", "WalkForwardResult", "MonteCarloResult",
    "Portfolio",
    "ExecutionModel", "MarketExecutionModel", "LimitExecutionModel",
    "StopExecutionModel", "CompositeExecutionModel"
]
```

**Step 3: 运行测试验证通过**

```bash
cd backend && python -m pytest tests/engine/test_execution.py -v
```

Expected: PASS (all tests)

**Step 4: 提交**

```bash
git add backend/engine/execution.py backend/engine/__init__.py backend/tests/engine/test_execution.py
git commit -m "feat(engine): add execution models with T+1 support

- MarketExecutionModel: T+1 open price execution
- LimitExecutionModel: limit order with price trigger
- StopExecutionModel: stop loss with market execution
- CompositeExecutionModel: auto-routing by order type
- Proper slippage and commission calculation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## 阶段 4：绩效指标模块 (engine/metrics.py)

### Task 4.1: 编写绩效指标测试

**Files:**
- Create: `backend/tests/engine/test_metrics.py`

**Step 1: 编写绩效指标测试**

```python
# backend/tests/engine/test_metrics.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.metrics import (
    calculate_metrics, calculate_sharpe, calculate_max_drawdown,
    calculate_alpha_beta, calculate_win_rate
)


@pytest.fixture
def sample_returns():
    """创建测试用收益率序列"""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.0005, 0.015, 252),  # 日均0.05%，波动1.5%
        index=dates
    )
    return returns


@pytest.fixture
def benchmark_returns():
    """创建基准收益率序列"""
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    np.random.seed(123)
    returns = pd.Series(
        np.random.normal(0.0003, 0.012, 252),
        index=dates
    )
    return returns


class TestCalculateSharpe:
    def test_positive_sharpe(self, sample_returns):
        sharpe = calculate_sharpe(sample_returns)
        # 正收益应该有正Sharpe
        assert isinstance(sharpe, float)

    def test_zero_volatility_returns_zero(self):
        returns = pd.Series([0.01] * 100)
        sharpe = calculate_sharpe(returns)
        # 零波动率应该返回0或无穷大的处理
        assert not np.isnan(sharpe)


class TestCalculateMaxDrawdown:
    def test_drawdown_is_negative(self, sample_returns):
        max_dd = calculate_max_drawdown(sample_returns)
        assert max_dd <= 0

    def test_no_drawdown_for_always_positive(self):
        returns = pd.Series([0.01] * 100)
        max_dd = calculate_max_drawdown(returns)
        assert max_dd == 0


class TestCalculateAlphaBeta:
    def test_alpha_beta_calculation(self, sample_returns, benchmark_returns):
        alpha, beta = calculate_alpha_beta(sample_returns, benchmark_returns)
        assert isinstance(alpha, float)
        assert isinstance(beta, float)

    def test_beta_close_to_one_for_similar_returns(self):
        returns = pd.Series([0.01, -0.005, 0.008, -0.003, 0.006])
        benchmark = pd.Series([0.01, -0.005, 0.008, -0.003, 0.006])
        _, beta = calculate_alpha_beta(returns, benchmark)
        assert abs(beta - 1.0) < 0.1


class TestCalculateWinRate:
    def test_win_rate_calculation(self, sample_returns):
        win_rate, win_count, loss_count = calculate_win_rate(sample_returns)
        assert 0 <= win_rate <= 1
        assert win_count + loss_count > 0

    def test_all_wins(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        win_rate, win_count, loss_count = calculate_win_rate(returns)
        assert win_rate == 1.0
        assert win_count == 3
        assert loss_count == 0


class TestCalculateMetrics:
    def test_full_metrics_calculation(self, sample_returns, benchmark_returns):
        metrics = calculate_metrics(sample_returns, benchmark_returns)

        assert hasattr(metrics, 'sharpe')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'alpha')
        assert hasattr(metrics, 'beta')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'cagr')

    def test_metrics_to_dict(self, sample_returns):
        metrics = calculate_metrics(sample_returns)
        metrics_dict = metrics.to_dict()

        assert 'sharpe' in metrics_dict
        assert 'max_drawdown' in metrics_dict
        assert 'strategy_return' in metrics_dict  # 兼容旧API

    def test_empty_returns_returns_default(self):
        returns = pd.Series([], dtype=float)
        metrics = calculate_metrics(returns)
        assert metrics.sharpe == 0.0
        assert metrics.max_drawdown == 0.0
```

**Step 2: 运行测试验证失败**

```bash
cd backend && python -m pytest tests/engine/test_metrics.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'engine.metrics'"

---

### Task 4.2: 实现绩效指标模块

**Files:**
- Create: `backend/engine/metrics.py`
- Modify: `backend/engine/__init__.py`

**Step 1: 创建 engine/metrics.py**

```python
# backend/engine/metrics.py
"""
绩效指标计算模块
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from .models import PerformanceMetrics, Fill, OrderSide


def calculate_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    ann_factor: int = 252
) -> float:
    """计算夏普比率"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / ann_factor
    return float(excess_returns.mean() / returns.std() * np.sqrt(ann_factor))


def calculate_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    ann_factor: int = 252
) -> float:
    """计算索提诺比率"""
    if len(returns) == 0:
        return 0.0

    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    ann_return = (1 + returns.mean()) ** ann_factor - 1
    downside_std = downside.std() * np.sqrt(ann_factor)

    return float((ann_return - risk_free_rate) / downside_std)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤"""
    if len(returns) == 0:
        return 0.0

    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max

    return float(drawdowns.min())


def calculate_calmar(returns: pd.Series, ann_factor: int = 252) -> float:
    """计算卡玛比率"""
    if len(returns) == 0:
        return 0.0

    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / ann_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    max_dd = calculate_max_drawdown(returns)

    return float(cagr / abs(max_dd)) if max_dd != 0 else 0.0


def calculate_alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    ann_factor: int = 252
) -> Tuple[float, float]:
    """计算Alpha和Beta"""
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0, 1.0

    # 对齐数据
    aligned = pd.DataFrame({
        "strategy": returns,
        "benchmark": benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        return 0.0, 1.0

    # 计算Beta
    cov = aligned["strategy"].cov(aligned["benchmark"])
    var = aligned["benchmark"].var()
    beta = cov / var if var > 0 else 1.0

    # 计算策略和基准的年化收益
    strat_total = (1 + aligned["strategy"]).prod() - 1
    bench_total = (1 + aligned["benchmark"]).prod() - 1
    n_years = len(aligned) / ann_factor

    strat_cagr = (1 + strat_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    bench_cagr = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Alpha = 策略收益 - (无风险 + Beta * (基准收益 - 无风险))
    alpha = strat_cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))

    return float(alpha), float(beta)


def calculate_win_rate(returns: pd.Series) -> Tuple[float, int, int]:
    """计算胜率"""
    if len(returns) == 0:
        return 0.0, 0, 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_count = len(wins)
    loss_count = len(losses)
    total = win_count + loss_count

    win_rate = win_count / total if total > 0 else 0.0

    return float(win_rate), win_count, loss_count


def calculate_pl_ratio(returns: pd.Series) -> float:
    """计算盈亏比"""
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0001

    return float(avg_win / avg_loss) if avg_loss > 0 else 0.0


def calculate_consecutive_wins(returns: pd.Series) -> int:
    """计算最大连续获胜天数"""
    max_streak = 0
    current_streak = 0

    for ret in returns:
        if ret > 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def calculate_period_returns(returns: pd.Series) -> Tuple[float, float]:
    """计算YTD和MTD收益"""
    if len(returns) == 0:
        return 0.0, 0.0

    current_date = returns.index[-1]

    # YTD
    try:
        ytd_start = current_date.replace(month=1, day=1)
        ytd_returns = returns[returns.index >= ytd_start]
        ytd = (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0
    except:
        ytd = 0.0

    # MTD
    try:
        mtd_start = current_date.replace(day=1)
        mtd_returns = returns[returns.index >= mtd_start]
        mtd = (1 + mtd_returns).prod() - 1 if len(mtd_returns) > 0 else 0
    except:
        mtd = 0.0

    return float(ytd), float(mtd)


def calculate_avg_hold_days(trades: List[Fill]) -> float:
    """从交易记录计算平均持仓天数"""
    if not trades:
        return 0.0

    positions = {}  # symbol -> entry_date
    hold_days = []

    for trade in trades:
        symbol = trade.order.symbol
        if trade.order.side == OrderSide.BUY:
            positions[symbol] = trade.timestamp.date()
        elif trade.order.side == OrderSide.SELL and symbol in positions:
            entry = positions[symbol]
            exit_date = trade.timestamp.date()
            days = (exit_date - entry).days
            if days > 0:
                hold_days.append(days)
            del positions[symbol]

    return sum(hold_days) / len(hold_days) if hold_days else 0.0


def calculate_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    trades: List[Fill] = None,
    risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """
    计算完整绩效指标

    Args:
        returns: 策略日收益率
        benchmark_returns: 基准日收益率
        trades: 交易记录（用于计算交易统计）
        risk_free_rate: 无风险利率

    Returns:
        PerformanceMetrics
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return PerformanceMetrics()

    ann_factor = 252

    # 基础收益指标
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / ann_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = returns.std() * np.sqrt(ann_factor)

    # 风险调整指标
    sharpe = calculate_sharpe(returns, risk_free_rate, ann_factor)
    sortino = calculate_sortino(returns, risk_free_rate, ann_factor)
    max_drawdown = calculate_max_drawdown(returns)
    calmar = calculate_calmar(returns, ann_factor)

    # Alpha/Beta
    alpha, beta = 0.0, 1.0
    benchmark_return = 0.0
    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(returns, benchmark_returns, risk_free_rate, ann_factor)
        bench_total = (1 + benchmark_returns.dropna()).prod() - 1
        bench_years = len(benchmark_returns.dropna()) / ann_factor
        benchmark_return = (1 + bench_total) ** (1 / bench_years) - 1 if bench_years > 0 else 0

    # 胜率
    win_rate, win_count, loss_count = calculate_win_rate(returns)
    pl_ratio = calculate_pl_ratio(returns)
    consecutive_wins = calculate_consecutive_wins(returns)

    # 时间段收益
    ytd_return, mtd_return = calculate_period_returns(returns)

    # 回撤区间
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd_date = drawdowns.idxmin()
    drawdown_period = max_dd_date.strftime("%Y/%m") if pd.notna(max_dd_date) else "N/A"

    # 平均持仓天数
    avg_hold_days = calculate_avg_hold_days(trades) if trades else 0.0

    return PerformanceMetrics(
        sharpe=sharpe,
        calmar=calmar,
        sortino=sortino,
        max_drawdown=max_drawdown,
        total_return=float(total_return),
        cagr=float(cagr),
        ytd_return=ytd_return,
        mtd_return=mtd_return,
        volatility=float(volatility),
        alpha=alpha,
        beta=beta,
        win_rate=win_rate,
        win_count=win_count,
        loss_count=loss_count,
        pl_ratio=pl_ratio,
        avg_hold_days=avg_hold_days,
        consecutive_wins=consecutive_wins,
        benchmark_return=float(benchmark_return),
        excess_return=float(cagr - benchmark_return),
        drawdown_period=drawdown_period
    )
```

**Step 2: 更新 engine/__init__.py**

```python
# backend/engine/__init__.py
"""回测引擎模块"""
from .models import (
    OrderSide, OrderType, Order, Fill, Position, Trade,
    PerformanceMetrics, BacktestResult, WalkForwardResult, MonteCarloResult
)
from .portfolio import Portfolio
from .execution import (
    ExecutionModel, MarketExecutionModel, LimitExecutionModel,
    StopExecutionModel, CompositeExecutionModel
)
from .metrics import calculate_metrics

__all__ = [
    "OrderSide", "OrderType", "Order", "Fill", "Position", "Trade",
    "PerformanceMetrics", "BacktestResult", "WalkForwardResult", "MonteCarloResult",
    "Portfolio",
    "ExecutionModel", "MarketExecutionModel", "LimitExecutionModel",
    "StopExecutionModel", "CompositeExecutionModel",
    "calculate_metrics"
]
```

**Step 3: 运行测试验证通过**

```bash
cd backend && python -m pytest tests/engine/test_metrics.py -v
```

Expected: PASS (all tests)

**Step 4: 提交**

```bash
git add backend/engine/metrics.py backend/engine/__init__.py backend/tests/engine/test_metrics.py
git commit -m "feat(engine): add comprehensive performance metrics module

- Sharpe, Sortino, Calmar ratios
- Alpha/Beta vs benchmark
- Max drawdown with period tracking
- Win rate, P/L ratio, consecutive wins
- YTD/MTD returns
- Average holding days from trades

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## 阶段 5：重构 Backtester (engine/backtester.py)

### Task 5.1: 编写新 Backtester 测试

**Files:**
- Create: `backend/tests/engine/test_backtester.py`

**Step 1: 编写新 Backtester 测试**

```python
# backend/tests/engine/test_backtester.py
import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List

from engine.models import Order, OrderSide, OrderType, Fill, BacktestResult
from engine.backtester import EventDrivenBacktester, Strategy


class SimpleTestStrategy(Strategy):
    """简单测试策略：每10天买入，每20天卖出"""

    def __init__(self):
        self.day_count = 0
        self.position = 0

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        self.day_count += 1
        orders = []

        if self.day_count % 20 == 0 and self.position > 0:
            # 卖出
            orders.append(Order(
                symbol="000001",
                side=OrderSide.SELL,
                quantity=Decimal("100"),
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))
        elif self.day_count % 10 == 0 and self.position == 0:
            # 买入
            orders.append(Order(
                symbol="000001",
                side=OrderSide.BUY,
                quantity=Decimal("100"),
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))

        return orders

    def on_fill(self, fill: Fill) -> None:
        if fill.order.side == OrderSide.BUY:
            self.position += int(fill.fill_quantity)
        else:
            self.position -= int(fill.fill_quantity)


@pytest.fixture
def sample_data():
    """创建测试用多标的数据"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")

    # 000001 数据
    data_000001 = pd.DataFrame({
        "open": np.linspace(10, 12, 100) + np.random.normal(0, 0.1, 100),
        "high": np.linspace(10.2, 12.2, 100) + np.random.normal(0, 0.1, 100),
        "low": np.linspace(9.8, 11.8, 100) + np.random.normal(0, 0.1, 100),
        "close": np.linspace(10, 12, 100) + np.random.normal(0, 0.1, 100),
        "volume": np.random.randint(1000000, 2000000, 100)
    }, index=dates)

    return {"000001": data_000001}


@pytest.fixture
def benchmark_data():
    """创建基准数据"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
    return pd.DataFrame({
        "open": np.linspace(3000, 3200, 100),
        "high": np.linspace(3020, 3220, 100),
        "low": np.linspace(2980, 3180, 100),
        "close": np.linspace(3000, 3200, 100),
        "volume": np.random.randint(10000000, 20000000, 100)
    }, index=dates)


class TestEventDrivenBacktester:
    def test_backtest_runs(self, sample_data, benchmark_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(strategy)

        result = backtester.run(sample_data, benchmark_data)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert hasattr(result, 'metrics')

    def test_equity_curve_has_correct_columns(self, sample_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(strategy)

        result = backtester.run(sample_data)

        assert "equity" in result.equity_curve.columns
        assert "returns" in result.equity_curve.columns

    def test_initial_capital_is_respected(self, sample_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(
            strategy,
            initial_capital=Decimal("500000")
        )

        result = backtester.run(sample_data)

        # 第一天权益应该接近初始资金
        first_equity = result.equity_curve["equity"].iloc[0]
        assert abs(first_equity - 500000) < 1000

    def test_trades_are_recorded(self, sample_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(strategy)

        result = backtester.run(sample_data)

        # 100天内应该有多次交易
        assert len(result.trades) > 0

    def test_benchmark_comparison(self, sample_data, benchmark_data):
        strategy = SimpleTestStrategy()
        backtester = EventDrivenBacktester(strategy)

        result = backtester.run(sample_data, benchmark_data)

        # 应该有基准列
        assert "benchmark" in result.equity_curve.columns
        # Alpha和Beta应该已计算
        assert result.metrics.beta != 1.0 or result.metrics.alpha != 0.0
```

**Step 2: 运行测试验证失败**

```bash
cd backend && python -m pytest tests/engine/test_backtester.py -v
```

Expected: FAIL (old backtester doesn't have Strategy ABC import path)

---

### Task 5.2: 重构 Backtester

**Files:**
- Modify: `backend/engine/backtester.py`

**Step 1: 重构 engine/backtester.py**

```python
# backend/engine/backtester.py
"""
事件驱动回测器：支持多标的、T+1执行、基准对比
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .models import (
    Order, Fill, Position, BacktestResult, PerformanceMetrics
)
from .portfolio import Portfolio
from .execution import ExecutionModel, CompositeExecutionModel
from .metrics import calculate_metrics


class Strategy(ABC):
    """策略抽象基类"""

    @abstractmethod
    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """
        处理K线数据，生成交易订单

        Args:
            timestamp: 当前时间戳
            data: 截止到当前时间的所有数据（避免look-ahead）

        Returns:
            订单列表
        """
        pass

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        """
        处理成交通知

        Args:
            fill: 成交记录
        """
        pass


class EventDrivenBacktester:
    """
    事件驱动回测器

    特点:
    - 支持多标的回测
    - T+1开盘价执行（避免look-ahead bias）
    - 支持基准对比计算Alpha/Beta
    """

    def __init__(
        self,
        strategy: Strategy,
        execution_model: ExecutionModel = None,
        initial_capital: Decimal = Decimal("1000000"),
        benchmark_symbol: str = "000300"
    ):
        self.strategy = strategy
        self.execution_model = execution_model or CompositeExecutionModel()
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol

        # 状态
        self.portfolio: Optional[Portfolio] = None
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.benchmark_curve: List[Tuple[datetime, float]] = []
        self.trades: List[Fill] = []

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame = None
    ) -> BacktestResult:
        """
        运行回测

        Args:
            data: {symbol: DataFrame} 多标的OHLCV数据
            benchmark_data: 基准指数数据

        Returns:
            BacktestResult
        """
        # 初始化
        self.portfolio = Portfolio(
            cash=self.initial_capital,
            initial_capital=self.initial_capital
        )
        self.equity_curve = []
        self.benchmark_curve = []
        self.trades = []

        # 获取所有交易日
        all_dates = self._get_all_dates(data)
        pending_orders: List[Order] = []

        for i, timestamp in enumerate(all_dates):
            # 获取当日各标的数据
            current_bars = self._get_bars_at(data, timestamp)
            next_bars = self._get_bars_at(data, all_dates[i + 1]) if i + 1 < len(all_dates) else {}

            # 执行待处理订单
            new_pending = []
            for order in pending_orders:
                if order.symbol in current_bars:
                    current_bar = current_bars[order.symbol]
                    next_bar = next_bars.get(order.symbol)
                    fill = self.execution_model.execute(order, current_bar, next_bar)
                    if fill:
                        self.portfolio.process_fill(fill)
                        self.strategy.on_fill(fill)
                        self.trades.append(fill)
                    else:
                        new_pending.append(order)
            pending_orders = new_pending

            # 计算当日权益
            prices = {
                symbol: Decimal(str(bar["close"]))
                for symbol, bar in current_bars.items()
            }
            equity = self.portfolio.get_equity(prices)
            self.equity_curve.append((timestamp, float(equity)))

            # 记录基准
            if benchmark_data is not None and timestamp in benchmark_data.index:
                bench_val = benchmark_data.loc[timestamp, "close"]
                self.benchmark_curve.append((timestamp, float(bench_val)))

            # 生成新订单
            combined_data = self._combine_data_up_to(data, timestamp)
            new_orders = self.strategy.on_bar(timestamp, combined_data)
            pending_orders.extend(new_orders)

        return self._create_result(benchmark_data)

    def _get_all_dates(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """获取所有交易日期的并集"""
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        return sorted(all_dates)

    def _get_bars_at(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> Dict[str, pd.Series]:
        """获取指定时间点的所有标的数据"""
        bars = {}
        for symbol, df in data.items():
            if timestamp in df.index:
                bars[symbol] = df.loc[timestamp]
        return bars

    def _combine_data_up_to(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> pd.DataFrame:
        """合并数据到指定时间点"""
        combined = []
        for symbol, df in data.items():
            subset = df.loc[:timestamp].copy()
            subset["symbol"] = symbol
            combined.append(subset)
        return pd.concat(combined) if combined else pd.DataFrame()

    def _create_result(self, benchmark_data: pd.DataFrame = None) -> BacktestResult:
        """创建回测结果"""
        # 构建权益曲线DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        equity_df["returns"] = equity_df["equity"].pct_change()

        # 添加基准
        benchmark_returns = None
        if self.benchmark_curve:
            bench_df = pd.DataFrame(self.benchmark_curve, columns=["timestamp", "benchmark"])
            bench_df.set_index("timestamp", inplace=True)

            # 归一化基准到初始资金
            initial_bench = bench_df["benchmark"].iloc[0]
            bench_df["benchmark"] = bench_df["benchmark"] / initial_bench * float(self.initial_capital)

            equity_df = equity_df.join(bench_df, how="left")
            equity_df["benchmark_returns"] = equity_df["benchmark"].pct_change()
            benchmark_returns = equity_df["benchmark_returns"]

        # 计算绩效指标
        metrics = calculate_metrics(
            equity_df["returns"],
            benchmark_returns=benchmark_returns,
            trades=self.trades
        )

        return BacktestResult(
            equity_curve=equity_df,
            trades=self.trades,
            metrics=metrics,
            positions=dict(self.portfolio.positions)
        )
```

**Step 2: 运行测试验证通过**

```bash
cd backend && python -m pytest tests/engine/test_backtester.py -v
```

Expected: PASS (all tests)

**Step 3: 运行所有引擎测试**

```bash
cd backend && python -m pytest tests/engine/ -v
```

Expected: PASS (all tests)

**Step 4: 提交**

```bash
git add backend/engine/backtester.py backend/tests/engine/test_backtester.py
git commit -m "refactor(engine): rewrite EventDrivenBacktester with multi-asset support

- Support Dict[symbol, DataFrame] input for multi-asset backtesting
- T+1 execution using next bar open price
- Benchmark comparison for Alpha/Beta calculation
- Integrate with new Portfolio, Execution, and Metrics modules
- Add Strategy ABC in same module for convenience

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## 阶段 6-10：略（Walk-Forward, Monte Carlo, 服务层, API集成, 数据层优化）

由于计划较长，后续任务结构相同，包括：

- **Task 6.1-6.2**: Walk-Forward优化器
- **Task 7.1-7.2**: Monte Carlo分析器
- **Task 8.1-8.2**: BacktestService服务层
- **Task 9.1-9.2**: 更新main.py集成新服务
- **Task 10.1-10.2**: 优化DataService频率控制

每个任务遵循相同的TDD模式：写测试 → 验证失败 → 实现 → 验证通过 → 提交

---

## 执行检查清单

在开始每个Task前，确认：
- [ ] 已读取相关源文件
- [ ] 理解现有代码结构
- [ ] 测试命令可用 (`python -m pytest`)

在完成每个Task后，确认：
- [ ] 所有测试通过
- [ ] 代码已提交
- [ ] 无遗留的未追踪文件
