# 事件驱动回测系统架构文档

## 概述

本系统是一个**事件驱动回测引擎**，支持多标的回测、T+1 开盘价执行、基准对比计算 Alpha/Beta。核心设计目标是避免 look-ahead bias（未来函数偏差）。

---

## 系统架构

### 模块结构

```
backend/engine/
├── models.py        # 核心数据模型
├── backtester.py    # 事件驱动回测器
├── execution.py     # 订单执行模型
├── portfolio.py     # 组合管理
├── metrics.py       # 绩效指标计算
├── data_loader.py   # 数据加载（AkShare / Mock）
├── walk_forward.py  # Walk-Forward 分析
└── monte_carlo.py   # Monte Carlo 模拟
```

### 核心组件关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                    EventDrivenBacktester                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Strategy   │  │  Portfolio   │  │   ExecutionModel     │  │
│  │  (用户实现)   │  │  (持仓管理)   │  │   (订单执行)          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
    on_bar()           process_fill()       execute()
    on_fill()          get_equity()
```

---

## 核心数据模型 (models.py)

### 订单相关

```python
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"   # 市价单
    LIMIT = "limit"     # 限价单
    STOP = "stop"       # 止损单

@dataclass
class Order:
    symbol: str                    # 标的代码
    side: OrderSide               # 买/卖方向
    quantity: Decimal             # 数量
    order_type: OrderType         # 订单类型
    limit_price: Optional[Decimal]  # 限价（限价单用）
    stop_price: Optional[Decimal]   # 止损价（止损单用）
    timestamp: Optional[datetime]   # 下单时间
```

### 成交记录

```python
@dataclass
class Fill:
    order: Order           # 原始订单
    fill_price: Decimal    # 成交价格
    fill_quantity: Decimal # 成交数量
    commission: Decimal    # 手续费
    slippage: Decimal      # 滑点金额
    timestamp: datetime    # 成交时间
```

### 持仓

```python
@dataclass
class Position:
    symbol: str                    # 标的代码
    quantity: Decimal              # 持仓数量
    avg_cost: Decimal              # 平均成本
    realized_pnl: Decimal          # 已实现盈亏
    entry_date: Optional[date]     # 建仓日期
```

### 回测结果

```python
@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame     # 权益曲线
    trades: List[Fill]             # 所有成交记录
    metrics: PerformanceMetrics    # 绩效指标
    positions: Dict[str, Position] # 最终持仓
```

---

## 回测主循环 (backtester.py)

### EventDrivenBacktester.run() 流程

```
输入: data (多标的OHLCV数据), benchmark_data (基准数据)

初始化阶段:
  1. 创建 Portfolio (初始资金)
  2. 清空 equity_curve, benchmark_curve, trades
  3. 获取所有交易日期的并集

主循环 (遍历每个交易日):
  ┌────────────────────────────────────────────────────────────┐
  │  for timestamp in all_dates:                               │
  │                                                            │
  │  Step 1: 获取当日数据                                       │
  │    current_bars = 当日各标的 OHLCV                          │
  │    next_bars = 下一交易日各标的 OHLCV (用于 T+1 执行)        │
  │                                                            │
  │  Step 2: 执行待处理订单 (来自前一日信号)                     │
  │    for order in pending_orders:                            │
  │      fill = execution_model.execute(order, current, next)  │
  │      if fill:                                              │
  │        portfolio.process_fill(fill)  # 更新持仓和现金       │
  │        strategy.on_fill(fill)        # 通知策略             │
  │        trades.append(fill)           # 记录成交             │
  │                                                            │
  │  Step 3: 计算当日权益                                       │
  │    prices = {symbol: close_price}                          │
  │    equity = portfolio.get_equity(prices)                   │
  │    equity_curve.append((timestamp, equity))                │
  │                                                            │
  │  Step 4: 记录基准                                          │
  │    if benchmark_data:                                      │
  │      benchmark_curve.append((timestamp, bench_close))      │
  │                                                            │
  │  Step 5: 生成新订单 (策略信号)                              │
  │    combined_data = 截止当日的所有历史数据                    │
  │    new_orders = strategy.on_bar(timestamp, combined_data)  │
  │    pending_orders.extend(new_orders)                       │
  │                                                            │
  └────────────────────────────────────────────────────────────┘

结束阶段:
  1. 构建 equity_curve DataFrame
  2. 计算收益率序列
  3. 调用 calculate_metrics() 计算绩效指标
  4. 返回 BacktestResult
```

### 关键设计：T+1 执行机制

```
T-1 日:
  - 策略 on_bar() 接收截止 T-1 的数据
  - 生成订单 (基于 T-1 收盘后的判断)
  - 订单进入 pending_orders

T 日:
  - 获取 T 日和 T+1 日的数据
  - 执行 pending_orders 中的订单
  - 市价单: 使用 T+1 日开盘价成交 (避免 look-ahead)
  - 策略基于 T 日数据生成新订单
```

**为什么用 T+1 开盘价？**
- 策略在 T 日收盘后计算信号
- 实际交易只能在 T+1 日开盘时执行
- 使用 T 日收盘价成交会引入未来函数偏差

---

## 订单执行模型 (execution.py)

### 执行模型层次

```
ExecutionModel (ABC)
    │
    ├── MarketExecutionModel     # 市价单
    ├── LimitExecutionModel      # 限价单
    ├── StopExecutionModel       # 止损单
    │
    └── CompositeExecutionModel  # 组合模型 (自动路由)
```

### MarketExecutionModel (市价单)

```python
执行逻辑:
  1. 检查订单类型是否为 MARKET
  2. 检查是否有 next_bar (T+1 数据)
  3. 使用 next_bar 的开盘价作为基础价格
  4. 应用滑点:
     - 买入: fill_price = open * (1 + slippage_bps/10000)
     - 卖出: fill_price = open / (1 + slippage_bps/10000)
  5. 计算佣金: max(trade_value * 0.0003, 5元)
  6. 返回 Fill

参数:
  - slippage_bps: 滑点基点 (默认 10bps = 0.1%)
  - commission_rate: 佣金费率 (默认万三)
```

### LimitExecutionModel (限价单)

```python
执行逻辑:
  1. 检查订单类型是否为 LIMIT
  2. 获取 K 线的最高/最低价
  3. 买入限价单: 最低价 <= 限价 则成交
  4. 卖出限价单: 最高价 >= 限价 则成交
  5. 以限价成交，无滑点
```

### StopExecutionModel (止损单)

```python
执行逻辑:
  1. 检查订单类型是否为 STOP
  2. 止损卖出: 最低价 <= 止损价 则触发
  3. 止损买入: 最高价 >= 止损价 则触发
  4. 触发后以开盘价 + 滑点成交
  5. 止损单滑点更大 (默认 20bps)
```

---

## 组合管理 (portfolio.py)

### Portfolio 类

```python
class Portfolio:
    cash: Decimal                      # 可用现金
    positions: Dict[str, Position]     # 持仓字典
    initial_capital: Decimal           # 初始资金

    def process_fill(fill):
        """处理成交"""
        position = get_position(fill.symbol)
        position.update(fill)  # 更新持仓

        if BUY:
            cash -= fill_price * quantity + commission
        else:
            cash += fill_price * quantity - commission

    def get_equity(prices):
        """计算总权益"""
        equity = cash
        for position in positions:
            equity += position.quantity * prices[symbol]
        return equity
```

### Position.update() 持仓更新逻辑

```python
def update(fill):
    if BUY:
        # 加仓: 更新平均成本
        new_qty = quantity + fill_quantity
        avg_cost = (quantity * avg_cost + fill_qty * fill_price) / new_qty
        quantity = new_qty

    else:  # SELL
        # 减仓: 计算已实现盈亏
        realized_pnl += fill_quantity * (fill_price - avg_cost)
        quantity -= fill_quantity
```

---

## 策略接口 (Strategy ABC)

### 策略必须实现的方法

```python
class Strategy(ABC):

    @abstractmethod
    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """
        每根 K 线调用一次

        Args:
            timestamp: 当前时间戳
            data: 截止当前的所有历史数据 (避免 look-ahead)

        Returns:
            订单列表
        """
        pass

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        """
        订单成交后调用

        Args:
            fill: 成交记录
        """
        pass
```

### 策略示例：AlphaTrendStrategy

```python
class AlphaTrendStrategy(Strategy):
    """
    趋势跟踪策略:
    - 入场: 短期均线上穿长期均线 (金叉)
    - 出场: 短期均线下穿长期均线 (死叉) 或 止损
    """

    def __init__(self, short_window=10, long_window=30, stop_loss_pct=0.05):
        self.short_window = short_window
        self.long_window = long_window
        self.stop_loss_pct = stop_loss_pct

        self.position_held = False
        self.position_entry_price = 0

    def on_bar(self, timestamp, data):
        orders = []

        # 计算均线
        closes = data["close"]
        short_ma = closes.rolling(self.short_window).mean()
        long_ma = closes.rolling(self.long_window).mean()

        # 检查止损
        if self.position_held:
            if current_price < entry_price * (1 - stop_loss_pct):
                orders.append(SELL_ORDER)
                return orders

        # 金叉买入
        if not position_held and 金叉信号:
            orders.append(BUY_ORDER)

        # 死叉卖出
        elif position_held and 死叉信号:
            orders.append(SELL_ORDER)

        return orders

    def on_fill(self, fill):
        if fill.order.side == BUY:
            self.position_held = True
            self.position_entry_price = fill.fill_price
        else:
            self.position_held = False
```

---

## 绩效指标计算 (metrics.py)

### 计算的指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Sharpe** | 夏普比率 | (年化收益 - 无风险) / 年化波动率 |
| **Sortino** | 索提诺比率 | (年化收益 - 无风险) / 下行波动率 |
| **Calmar** | 卡玛比率 | CAGR / \|MaxDD\| |
| **MaxDD** | 最大回撤 | (峰值 - 谷值) / 峰值 |
| **CAGR** | 年化复合收益 | (终值/初值)^(1/年数) - 1 |
| **Alpha** | 超额收益 | 策略收益 - CAPM预期收益 |
| **Beta** | 市场敏感度 | Cov(策略, 基准) / Var(基准) |
| **Win Rate** | 胜率 | 盈利天数 / 总交易天数 |
| **P/L Ratio** | 盈亏比 | 平均盈利 / 平均亏损 |

### calculate_metrics() 主函数

```python
def calculate_metrics(returns, benchmark_returns, trades):
    """
    Args:
        returns: 策略日收益率序列
        benchmark_returns: 基准日收益率序列
        trades: 成交记录列表

    Returns:
        PerformanceMetrics 对象
    """

    # 1. 基础收益指标
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (1/n_years) - 1
    volatility = returns.std() * sqrt(252)

    # 2. 风险调整指标
    sharpe = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    max_drawdown = calculate_max_drawdown(returns)
    calmar = calculate_calmar(returns)

    # 3. 相对基准指标
    alpha, beta = calculate_alpha_beta(returns, benchmark_returns)

    # 4. 交易统计
    win_rate, win_count, loss_count = calculate_win_rate(returns)
    pl_ratio = calculate_pl_ratio(returns)
    avg_hold_days = calculate_avg_hold_days(trades)

    return PerformanceMetrics(...)
```

---

## 数据加载 (data_loader.py)

### 数据源

1. **AkShare API** - 真实 A 股数据
2. **Mock 数据** - 用于演示和测试

### 数据格式要求

```python
DataFrame 结构:
  Index: DatetimeIndex (交易日期)
  Columns:
    - open: 开盘价
    - high: 最高价
    - low: 最低价
    - close: 收盘价
    - volume: 成交量
    - symbol: 标的代码
```

### Mock 数据生成

```python
def generate_mock_data(start_date, days, symbol):
    """
    使用几何布朗运动 (GBM) 生成模拟价格

    参数:
      - mu = 0.08 (年化漂移 8%)
      - sigma = 0.25 (年化波动率 25%)
    """
```

---

## 完整回测流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                        回测启动                                   │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. 加载数据                                                      │
│     - fetch_stock_data() 或 generate_mock_data()                 │
│     - 格式: {symbol: DataFrame}                                   │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. 初始化回测器                                                  │
│     - EventDrivenBacktester(strategy, execution_model, capital)  │
│     - Portfolio(cash=initial_capital)                            │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. 运行回测 backtester.run(data, benchmark)                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  遍历每个交易日 T:                                          │  │
│  │                                                            │  │
│  │  [执行阶段]                                                 │  │
│  │    pending_orders → ExecutionModel → Fill                  │  │
│  │    Fill → Portfolio.process_fill()                         │  │
│  │    Fill → Strategy.on_fill()                               │  │
│  │                                                            │  │
│  │  [估值阶段]                                                 │  │
│  │    Portfolio.get_equity(current_prices)                    │  │
│  │    → equity_curve.append()                                 │  │
│  │                                                            │  │
│  │  [信号阶段]                                                 │  │
│  │    Strategy.on_bar(T, data[:T])                            │  │
│  │    → new_orders → pending_orders                           │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. 计算绩效指标                                                  │
│     - calculate_metrics(returns, benchmark_returns, trades)      │
│     - 输出: Sharpe, MaxDD, CAGR, Alpha, Beta, Win Rate...       │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. 返回 BacktestResult                                          │
│     - equity_curve: 权益曲线                                     │
│     - trades: 所有成交记录                                        │
│     - metrics: 绩效指标                                          │
│     - positions: 最终持仓                                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 如何添加新策略

1. 创建策略文件 `backend/strategies/your_strategy.py`

2. 继承 Strategy 基类并实现两个方法:

```python
from engine.backtester import Strategy
from engine.models import Order, OrderSide, OrderType, Fill

class YourStrategy(Strategy):
    def __init__(self, param1, param2):
        # 初始化参数和状态
        pass

    def on_bar(self, timestamp, data) -> List[Order]:
        # 计算信号
        # 返回订单列表
        return []

    def on_fill(self, fill) -> None:
        # 更新内部状态
        pass
```

3. 在 `main.py` 中注册策略

4. 前端自动通过 `/api/strategies` 接口获取

---

## 注意事项

### 避免 Look-Ahead Bias

- `on_bar()` 只接收截止当前时间的数据
- 市价单在 T+1 开盘价执行，而非当日收盘价
- 指标计算使用收盘后数据

### 佣金和滑点

- 默认佣金: 万三 (0.03%)，最低 5 元
- 默认滑点: 市价单 10bps，止损单 20bps

### 多标的支持

- 数据格式: `Dict[str, DataFrame]`
- Portfolio 自动管理多标的持仓
- 策略可同时生成多标的订单
