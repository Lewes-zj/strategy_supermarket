# 回测系统重构设计文档

> 日期: 2026-01-30
> 状态: 待实施
> 目标: 基于 SKILL.md 框架完成完整回测系统开发

---

## 1. 项目背景

### 1.1 当前状态

策略超市项目已有基础回测功能，但存在以下问题：

| 问题 | 严重程度 | 说明 |
|------|----------|------|
| 执行价格用 close 而非 open | 高 | 造成 look-ahead bias |
| 单标的限制 | 高 | 无法做组合回测 |
| LIMIT/STOP 订单未实现 | 中 | 仅 MARKET 订单可用 |
| Alpha/Beta 硬编码 | 中 | 未与真实基准计算 |
| 缺少 Walk-Forward | 低 | 无法验证参数稳定性 |
| 缺少 Monte Carlo | 低 | 无法评估风险置信区间 |

### 1.2 AKShare 接口现状

当前调用的接口：

| 接口 | 用途 |
|------|------|
| `ak.stock_zh_a_hist()` | A股历史日线数据 (OHLCV) |
| `ak.stock_zh_a_spot_em()` | 实时行情 |

数据字段：日期、开盘、最高、最低、收盘、成交量、成交额，支持前复权/后复权。

---

## 2. 设计目标

1. **修复现有 bug** - 执行价格、多标的支持
2. **完善回测引擎** - 实现 SKILL.md 四种模式
3. **保持 API 兼容** - 前端无需改动
4. **真实数据** - 不使用 mock 数据

---

## 3. 系统架构

### 3.1 目录结构

```
backend/
├── engine/                          # 回测引擎核心
│   ├── __init__.py
│   ├── backtester.py               # 重构：事件驱动回测器
│   ├── vectorized.py               # 新增：向量化回测器
│   ├── walk_forward.py             # 新增：Walk-Forward优化器
│   ├── monte_carlo.py              # 新增：Monte Carlo分析
│   ├── execution.py                # 新增：执行模型（MARKET/LIMIT/STOP）
│   ├── portfolio.py                # 新增：组合管理（多标的）
│   ├── metrics.py                  # 新增：绩效指标计算（统一）
│   ├── models.py                   # 新增：数据模型
│   └── data_loader.py              # 保留：数据加载
│
├── services/
│   ├── data_service.py             # 优化：AKShare频率控制
│   ├── backtest_service.py         # 新增：回测服务层（统一调度）
│   └── ...
│
└── main.py                          # 保持API不变
```

### 3.2 模块关系图

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py (API)                        │
│   GET /metrics  GET /equity_curve  GET /transactions  ...   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BacktestService                          │
│   run_backtest()  run_walk_forward()  run_monte_carlo()     │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ EventDriven   │   │ WalkForward     │   │ MonteCarlo      │
│ Backtester    │   │ Optimizer       │   │ Analyzer        │
└───────┬───────┘   └─────────────────┘   └─────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Components                          │
│   Portfolio    ExecutionModel    Strategy    Metrics         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      DataService                             │
│   MySQL (历史数据)  ←──  AKShare (增量更新)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 核心模块设计

### 4.1 Portfolio - 多标的组合管理

```python
# engine/portfolio.py

@dataclass
class Position:
    symbol: str
    quantity: Decimal = Decimal("0")
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    entry_date: Optional[date] = None

    def update(self, fill: Fill) -> None:
        """更新持仓"""
        if fill.order.side == OrderSide.BUY:
            new_quantity = self.quantity + fill.fill_quantity
            if new_quantity != 0:
                self.avg_cost = (
                    (self.quantity * self.avg_cost + fill.fill_quantity * fill.fill_price)
                    / new_quantity
                )
            self.quantity = new_quantity
            if self.entry_date is None:
                self.entry_date = fill.timestamp.date()
        else:
            self.realized_pnl += fill.fill_quantity * (fill.fill_price - self.avg_cost)
            self.quantity -= fill.fill_quantity
            if self.quantity == 0:
                self.entry_date = None

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
```

### 4.2 ExecutionModel - 完整执行模型

```python
# engine/execution.py

class ExecutionModel(ABC):
    @abstractmethod
    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        """执行订单，返回成交结果"""
        pass

class MarketExecutionModel(ExecutionModel):
    """市价单执行：T日信号 → T+1开盘执行"""

    def __init__(self, slippage_bps: float = 10, commission_rate: float = 0.0003):
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate  # 万三佣金

    def execute(
        self,
        order: Order,
        current_bar: pd.Series,
        next_bar: Optional[pd.Series] = None
    ) -> Optional[Fill]:
        if order.order_type != OrderType.MARKET:
            return None
        if next_bar is None:
            return None  # 无下一根K线，无法执行

        # 使用下一根K线的开盘价（T+1执行）
        base_price = Decimal(str(next_bar["open"]))

        # 应用滑点
        slippage_mult = Decimal(str(1 + self.slippage_bps / 10000))
        if order.side == OrderSide.BUY:
            fill_price = base_price * slippage_mult
        else:
            fill_price = base_price / slippage_mult

        # 计算佣金
        trade_value = fill_price * order.quantity
        commission = trade_value * Decimal(str(self.commission_rate))
        commission = max(commission, Decimal("5"))  # 最低5元

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=abs(fill_price - base_price) * order.quantity,
            timestamp=next_bar.name if hasattr(next_bar, 'name') else datetime.now()
        )

class LimitExecutionModel(ExecutionModel):
    """限价单执行：价格触及才成交"""

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

        trade_value = fill_price * order.quantity
        commission = max(trade_value * Decimal(str(self.commission_rate)), Decimal("5"))

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=Decimal("0"),
            timestamp=bar.name if hasattr(bar, 'name') else datetime.now()
        )

class StopExecutionModel(ExecutionModel):
    """止损单执行：突破触发"""

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

        # 触发后以市价成交，带滑点
        slippage_mult = Decimal(str(1 + self.slippage_bps / 10000))
        if order.side == OrderSide.BUY:
            fill_price = open_price * slippage_mult
        else:
            fill_price = open_price / slippage_mult

        trade_value = fill_price * order.quantity
        commission = max(trade_value * Decimal(str(self.commission_rate)), Decimal("5"))

        return Fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission,
            slippage=abs(fill_price - open_price) * order.quantity,
            timestamp=bar.name if hasattr(bar, 'name') else datetime.now()
        )

class CompositeExecutionModel(ExecutionModel):
    """组合执行模型：根据订单类型选择执行器"""

    def __init__(self):
        self.market_model = MarketExecutionModel()
        self.limit_model = LimitExecutionModel()
        self.stop_model = StopExecutionModel()

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

### 4.3 EventDrivenBacktester - 重构后的事件驱动回测器

```python
# engine/backtester.py

class EventDrivenBacktester:
    """事件驱动回测器：支持多标的、T+1执行、基准对比"""

    def __init__(
        self,
        strategy: Strategy,
        execution_model: ExecutionModel = None,
        initial_capital: Decimal = Decimal("1000000"),
        benchmark_symbol: str = "000300"
    ):
        self.strategy = strategy
        self.execution_model = execution_model or CompositeExecutionModel()
        self.portfolio = Portfolio(cash=initial_capital, initial_capital=initial_capital)
        self.benchmark_symbol = benchmark_symbol
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.benchmark_curve: List[Tuple[datetime, float]] = []
        self.trades: List[Fill] = []

    def run(self, data: Dict[str, pd.DataFrame], benchmark_data: pd.DataFrame = None) -> BacktestResult:
        """
        运行回测

        Args:
            data: {symbol: DataFrame} 多标的OHLCV数据
            benchmark_data: 基准指数数据

        Returns:
            BacktestResult 包含权益曲线、交易记录、绩效指标
        """
        # 合并所有日期
        all_dates = self._get_all_dates(data)
        pending_orders: List[Order] = []

        for i, timestamp in enumerate(all_dates):
            # 获取当日各标的数据
            current_bars = self._get_bars_at(data, timestamp)
            next_bars = self._get_bars_at(data, all_dates[i + 1]) if i + 1 < len(all_dates) else {}

            # 执行待处理订单（使用下一根K线开盘价）
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
                        new_pending.append(order)  # 未成交，继续挂单
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

    def _get_bars_at(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> Dict[str, pd.Series]:
        """获取指定时间点的所有标的数据"""
        bars = {}
        for symbol, df in data.items():
            if timestamp in df.index:
                bars[symbol] = df.loc[timestamp]
        return bars

    def _combine_data_up_to(self, data: Dict[str, pd.DataFrame], timestamp: datetime) -> pd.DataFrame:
        """合并数据到指定时间点（避免look-ahead）"""
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
        if self.benchmark_curve:
            bench_df = pd.DataFrame(self.benchmark_curve, columns=["timestamp", "benchmark"])
            bench_df.set_index("timestamp", inplace=True)
            # 归一化基准到初始资金
            initial_bench = bench_df["benchmark"].iloc[0]
            bench_df["benchmark"] = bench_df["benchmark"] / initial_bench * float(self.portfolio.initial_capital)
            equity_df = equity_df.join(bench_df, how="left")
            equity_df["benchmark_returns"] = equity_df["benchmark"].pct_change()

        # 计算绩效指标
        metrics = calculate_metrics(
            equity_df["returns"],
            benchmark_returns=equity_df.get("benchmark_returns")
        )

        return BacktestResult(
            equity_curve=equity_df,
            trades=self.trades,
            metrics=metrics,
            positions=dict(self.portfolio.positions)
        )
```

### 4.4 WalkForwardOptimizer - Walk-Forward优化器

```python
# engine/walk_forward.py

class WalkForwardOptimizer:
    """Walk-Forward分析：滚动窗口优化验证参数稳定性"""

    def __init__(
        self,
        train_days: int = 252,      # 训练期1年
        test_days: int = 63,        # 测试期1季度
        step_days: int = 63,        # 滚动步长
        anchored: bool = False      # 是否锚定起点
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.anchored = anchored

    def optimize(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_factory: Callable[[Dict], Strategy],
        param_grid: Dict[str, List],
        metric: str = "sharpe_ratio",
        benchmark_data: pd.DataFrame = None
    ) -> WalkForwardResult:
        """
        运行Walk-Forward优化

        Args:
            data: 多标的数据
            strategy_factory: 参数字典 -> Strategy实例的工厂函数
            param_grid: 参数搜索空间 {"param_name": [values]}
            metric: 优化目标指标
            benchmark_data: 基准数据

        Returns:
            WalkForwardResult
        """
        splits = self._generate_splits(data)
        all_results = []
        param_history = []

        for i, (train_data, test_data) in enumerate(splits):
            # 在训练集上寻找最优参数
            best_params, best_metric = self._grid_search(
                train_data, strategy_factory, param_grid, metric, benchmark_data
            )
            param_history.append(best_params)

            # 在测试集上验证
            strategy = strategy_factory(best_params)
            backtester = EventDrivenBacktester(strategy)
            test_result = backtester.run(test_data, benchmark_data)
            test_result.split_index = i
            test_result.optimal_params = best_params
            all_results.append(test_result)

            print(f"Split {i+1}/{len(splits)}: {metric}={best_metric:.4f}, params={best_params}")

        # 计算参数稳定性评分
        stability_score = self._calculate_stability(param_history, param_grid)

        # 拼接测试期权益曲线
        combined_equity = self._combine_equity_curves(all_results)

        return WalkForwardResult(
            combined_equity=combined_equity,
            split_results=all_results,
            param_history=param_history,
            stability_score=stability_score
        )

    def _generate_splits(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[Dict, Dict]]:
        """生成训练/测试数据分割"""
        # 获取公共日期范围
        all_dates = self._get_common_dates(data)
        n = len(all_dates)
        splits = []

        start = 0
        while start + self.train_days + self.test_days <= n:
            train_start = 0 if self.anchored else start
            train_end = start + self.train_days
            test_end = min(train_end + self.test_days, n)

            train_dates = all_dates[train_start:train_end]
            test_dates = all_dates[train_end:test_end]

            train_data = {sym: df.loc[train_dates[0]:train_dates[-1]] for sym, df in data.items()}
            test_data = {sym: df.loc[test_dates[0]:test_dates[-1]] for sym, df in data.items()}

            splits.append((train_data, test_data))
            start += self.step_days

        return splits

    def _get_common_dates(self, data: Dict[str, pd.DataFrame]) -> List:
        """获取所有标的的公共日期"""
        date_sets = [set(df.index) for df in data.values()]
        common = set.intersection(*date_sets) if date_sets else set()
        return sorted(common)

    def _grid_search(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_factory: Callable,
        param_grid: Dict[str, List],
        metric: str,
        benchmark_data: pd.DataFrame
    ) -> Tuple[Dict, float]:
        """网格搜索最优参数"""
        from itertools import product

        best_params = None
        best_metric = float("-inf")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))
            strategy = strategy_factory(params)
            backtester = EventDrivenBacktester(strategy)
            result = backtester.run(data, benchmark_data)

            metric_value = getattr(result.metrics, metric, 0)
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params

        return best_params, best_metric

    def _calculate_stability(self, param_history: List[Dict], param_grid: Dict) -> float:
        """计算参数稳定性评分 (0-1)"""
        if len(param_history) < 2:
            return 1.0

        stability_scores = []
        for param_name in param_grid.keys():
            values = [p[param_name] for p in param_history]
            unique_ratio = len(set(values)) / len(values)
            stability_scores.append(1 - unique_ratio)

        return sum(stability_scores) / len(stability_scores) if stability_scores else 1.0

    def _combine_equity_curves(self, results: List[BacktestResult]) -> pd.DataFrame:
        """拼接各测试期的权益曲线"""
        curves = [r.equity_curve for r in results]
        return pd.concat(curves).sort_index()
```

### 4.5 MonteCarloAnalyzer - Monte Carlo分析

```python
# engine/monte_carlo.py

class MonteCarloAnalyzer:
    """Monte Carlo模拟分析：评估策略风险和收益置信区间"""

    def __init__(self, n_simulations: int = 1000, confidence: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence = confidence

    def analyze(self, returns: pd.Series) -> MonteCarloResult:
        """
        完整Monte Carlo分析

        Args:
            returns: 历史收益率序列

        Returns:
            MonteCarloResult
        """
        # Bootstrap模拟
        simulations = self._bootstrap_returns(returns)

        # 分析最大回撤分布
        drawdown_stats = self._analyze_drawdowns(simulations)

        # 计算VaR和CVaR
        var_95, cvar_95 = self._calculate_var(simulations)

        # 各持有期亏损概率
        prob_loss = self._probability_of_loss(returns, [21, 63, 126, 252])

        # 收益置信区间
        confidence_interval = self._return_confidence_interval(simulations)

        return MonteCarloResult(
            expected_max_drawdown=drawdown_stats["expected"],
            var_95=var_95,
            cvar_95=cvar_95,
            probability_of_loss=prob_loss,
            return_confidence_interval=confidence_interval,
            simulations=simulations
        )

    def _bootstrap_returns(self, returns: pd.Series, n_periods: int = None) -> np.ndarray:
        """Bootstrap重采样模拟"""
        if n_periods is None:
            n_periods = len(returns)

        simulations = np.zeros((self.n_simulations, n_periods))
        returns_arr = returns.dropna().values

        for i in range(self.n_simulations):
            simulations[i] = np.random.choice(returns_arr, size=n_periods, replace=True)

        return simulations

    def _analyze_drawdowns(self, simulations: np.ndarray) -> Dict[str, float]:
        """分析回撤分布"""
        max_drawdowns = []

        for sim_returns in simulations:
            equity = (1 + sim_returns).cumprod()
            rolling_max = np.maximum.accumulate(equity)
            drawdowns = (equity - rolling_max) / rolling_max
            max_drawdowns.append(drawdowns.min())

        max_drawdowns = np.array(max_drawdowns)

        return {
            "expected": float(np.mean(max_drawdowns)),
            "median": float(np.median(max_drawdowns)),
            "worst_95pct": float(np.percentile(max_drawdowns, 5)),
            "worst_case": float(max_drawdowns.min())
        }

    def _calculate_var(self, simulations: np.ndarray) -> Tuple[float, float]:
        """计算VaR和CVaR"""
        total_returns = (1 + simulations).prod(axis=1) - 1
        var_95 = float(np.percentile(total_returns, 5))
        cvar_95 = float(total_returns[total_returns <= var_95].mean())
        return var_95, cvar_95

    def _probability_of_loss(
        self,
        returns: pd.Series,
        holding_periods: List[int]
    ) -> Dict[int, float]:
        """计算各持有期亏损概率"""
        results = {}

        for period in holding_periods:
            if period > len(returns):
                continue
            simulations = self._bootstrap_returns(returns, period)
            total_returns = (1 + simulations).prod(axis=1) - 1
            prob_loss = float((total_returns < 0).mean())
            results[period] = prob_loss

        return results

    def _return_confidence_interval(self, simulations: np.ndarray) -> Tuple[float, float]:
        """计算收益置信区间"""
        total_returns = (1 + simulations).prod(axis=1) - 1
        lower = (1 - self.confidence) / 2 * 100
        upper = (1 - lower / 100) * 100
        return (
            float(np.percentile(total_returns, lower)),
            float(np.percentile(total_returns, upper))
        )

    def stress_test(
        self,
        returns: pd.Series,
        scenarios: Dict[str, Tuple[str, str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        历史情景压力测试

        Args:
            returns: 历史收益序列
            scenarios: 情景定义 {"名称": ("开始日期", "结束日期")}
        """
        if scenarios is None:
            scenarios = {
                "2008金融危机": ("2008-01-01", "2008-12-31"),
                "2015股灾": ("2015-06-01", "2015-09-30"),
                "2020疫情": ("2020-01-01", "2020-03-31"),
                "2022熊市": ("2022-01-01", "2022-10-31")
            }

        results = {}
        for name, (start, end) in scenarios.items():
            try:
                scenario_returns = returns.loc[start:end]
                if len(scenario_returns) > 0:
                    total_return = (1 + scenario_returns).prod() - 1
                    max_dd = self._calculate_max_drawdown(scenario_returns)
                    results[name] = {
                        "total_return": float(total_return),
                        "max_drawdown": float(max_dd),
                        "days": len(scenario_returns)
                    }
            except:
                continue

        return results

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        return drawdowns.min()
```

### 4.6 Metrics - 统一绩效指标计算

```python
# engine/metrics.py

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

    # Sharpe
    excess_return_series = returns - risk_free_rate / ann_factor
    sharpe = excess_return_series.mean() / returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0

    # Sortino
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 0.0001
    sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0

    # 最大回撤
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Calmar
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Alpha/Beta（真实计算）
    alpha, beta = 0.0, 1.0
    benchmark_return = 0.0
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        aligned = pd.DataFrame({"strategy": returns, "benchmark": benchmark_returns}).dropna()
        if len(aligned) > 1:
            cov = aligned["strategy"].cov(aligned["benchmark"])
            var = aligned["benchmark"].var()
            beta = cov / var if var > 0 else 1.0

            bench_total = (1 + aligned["benchmark"]).prod() - 1
            bench_cagr = (1 + bench_total) ** (ann_factor / len(aligned)) - 1 if len(aligned) > 0 else 0
            alpha = cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))
            benchmark_return = bench_cagr

    # 胜率
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_count = len(wins)
    loss_count = len(losses)
    total_trades = win_count + loss_count
    win_rate = win_count / total_trades if total_trades > 0 else 0

    # 盈亏比
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0001
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 连续获胜天数
    consecutive_wins = _calculate_consecutive_wins(returns)

    # YTD/MTD
    ytd_return, mtd_return = _calculate_period_returns(returns)

    # 回撤区间
    max_dd_date = drawdowns.idxmin()
    drawdown_period = max_dd_date.strftime("%Y/%m") if pd.notna(max_dd_date) else "N/A"

    # 平均持仓天数（从trades计算）
    avg_hold_days = _calculate_avg_hold_days(trades) if trades else 0

    return PerformanceMetrics(
        sharpe=float(sharpe),
        calmar=float(calmar),
        sortino=float(sortino),
        max_drawdown=float(max_drawdown),
        total_return=float(total_return),
        cagr=float(cagr),
        ytd_return=float(ytd_return),
        mtd_return=float(mtd_return),
        volatility=float(volatility),
        alpha=float(alpha),
        beta=float(beta),
        win_rate=float(win_rate),
        win_count=win_count,
        loss_count=loss_count,
        pl_ratio=float(pl_ratio),
        avg_hold_days=float(avg_hold_days),
        consecutive_wins=consecutive_wins,
        benchmark_return=float(benchmark_return),
        excess_return=float(cagr - benchmark_return),
        drawdown_period=drawdown_period
    )


def _calculate_consecutive_wins(returns: pd.Series) -> int:
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


def _calculate_period_returns(returns: pd.Series) -> Tuple[float, float]:
    """计算YTD和MTD收益"""
    if len(returns) == 0:
        return 0.0, 0.0

    current_date = returns.index[-1]

    # YTD
    ytd_start = current_date.replace(month=1, day=1)
    ytd_returns = returns[returns.index >= ytd_start]
    ytd = (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0

    # MTD
    mtd_start = current_date.replace(day=1)
    mtd_returns = returns[returns.index >= mtd_start]
    mtd = (1 + mtd_returns).prod() - 1 if len(mtd_returns) > 0 else 0

    return float(ytd), float(mtd)


def _calculate_avg_hold_days(trades: List[Fill]) -> float:
    """从交易记录计算平均持仓天数"""
    if not trades:
        return 0.0

    # 按symbol分组，计算每次完整交易的持仓天数
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
            hold_days.append(days)
            del positions[symbol]

    return sum(hold_days) / len(hold_days) if hold_days else 0.0
```

### 4.7 数据模型

```python
# engine/models.py

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd

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
class BacktestResult:
    """回测结果"""
    equity_curve: pd.DataFrame
    trades: List[Fill]
    metrics: 'PerformanceMetrics'
    positions: Dict[str, 'Position']
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
    simulations: Optional[any] = None  # numpy array
```

---

## 5. AKShare 频率控制

### 5.1 自适应限流器

```python
# services/rate_limiter.py

class AdaptiveRateLimiter:
    """自适应限流器"""

    def __init__(
        self,
        base_rate: float = 1.0,
        min_rate: float = 0.2,
        max_rate: float = 2.0,
        burst_size: int = 10
    ):
        self.base_rate = base_rate
        self.current_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.consecutive_success = 0
        self.consecutive_fail = 0
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """获取令牌"""
        with self._lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            if not blocking:
                return False

        # 阻塞等待
        start = time.time()
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            if timeout and (time.time() - start) > timeout:
                return False
            time.sleep(0.1)

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.current_rate
        self.tokens = min(self.burst_size, self.tokens + new_tokens)
        self.last_update = now

    def on_success(self):
        """请求成功回调"""
        self.consecutive_success += 1
        self.consecutive_fail = 0
        if self.consecutive_success >= 10:
            self.current_rate = min(self.current_rate * 1.2, self.max_rate)
            self.consecutive_success = 0

    def on_fail(self):
        """请求失败回调"""
        self.consecutive_fail += 1
        self.consecutive_success = 0
        self.current_rate = max(self.current_rate * 0.5, self.min_rate)

        # 指数退避
        backoff = min(2 ** self.consecutive_fail, 60)
        time.sleep(backoff)
```

### 5.2 数据获取策略

| 场景 | 频率 | 说明 |
|------|------|------|
| 初始化历史数据 | 0.5 req/s | 一次性，10年数据，夜间执行 |
| 每日增量更新 | 1.0 req/s | 15:30后执行，只拉当日 |
| 按需获取 | 优先缓存 | 缺失才拉取 |
| 被限流时 | 0.2 req/s | 自动降级 + 退避 |

---

## 6. 服务层设计

### 6.1 BacktestService

```python
# services/backtest_service.py

class BacktestService:
    """回测服务层：统一调度"""

    def __init__(self):
        self.data_service = get_data_service()
        self.cache = BacktestCache(ttl_hours=24)

    def run_backtest(
        self,
        strategy_id: str,
        symbols: List[str] = None,
        start_date: date = None,
        end_date: date = None
    ) -> BacktestResult:
        """标准回测"""
        # 默认10年数据
        if start_date is None:
            start_date = date.today() - timedelta(days=365 * 10)
        if end_date is None:
            end_date = date.today()

        # 检查缓存
        cache_key = f"{strategy_id}:{start_date}:{end_date}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # 加载数据（从数据库，不调用AKShare）
        strategy_info = get_strategy_info(strategy_id)
        if symbols is None:
            symbols = strategy_info.default_symbols

        data = self.data_service.get_data_for_backtest(symbols, start_date, end_date)
        benchmark = self.data_service.get_benchmark_data("000300", start_date, end_date)

        # 执行回测
        strategy = create_strategy(strategy_id)
        backtester = EventDrivenBacktester(strategy, benchmark_symbol="000300")
        result = backtester.run(data, benchmark)

        # 缓存
        self.cache.set(cache_key, result)
        return result

    def run_walk_forward(
        self,
        strategy_id: str,
        param_grid: Dict[str, List],
        train_days: int = 252,
        test_days: int = 63
    ) -> WalkForwardResult:
        """Walk-Forward分析"""
        strategy_info = get_strategy_info(strategy_id)
        symbols = strategy_info.default_symbols

        # 加载10年数据
        start_date = date.today() - timedelta(days=365 * 10)
        data = self.data_service.get_data_for_backtest(symbols, start_date, date.today())
        benchmark = self.data_service.get_benchmark_data("000300", start_date, date.today())

        # 创建策略工厂
        def strategy_factory(params: Dict) -> Strategy:
            return create_strategy(strategy_id, **params)

        optimizer = WalkForwardOptimizer(train_days, test_days)
        return optimizer.optimize(data, strategy_factory, param_grid, benchmark_data=benchmark)

    def run_monte_carlo(
        self,
        strategy_id: str,
        n_simulations: int = 1000
    ) -> MonteCarloResult:
        """Monte Carlo分析"""
        result = self.run_backtest(strategy_id)
        analyzer = MonteCarloAnalyzer(n_simulations)
        return analyzer.analyze(result.equity_curve["returns"])
```

---

## 7. API 兼容性

现有 API 端点完全不变，内部调用 BacktestService：

```python
# main.py 改动示例

from services.backtest_service import BacktestService

backtest_service = BacktestService()

@app.get("/api/strategies/{id}/metrics")
def get_metrics(id: str, year: Optional[int] = None):
    result = backtest_service.run_backtest(id)
    metrics = result.metrics
    # 转换为现有API格式
    return {
        "sharpe": metrics.sharpe,
        "calmar": metrics.calmar,
        # ... 其他字段保持不变
    }
```

---

## 8. 实施计划

### 阶段 1：核心引擎重构
- [ ] 创建 engine/models.py
- [ ] 创建 engine/portfolio.py
- [ ] 创建 engine/execution.py
- [ ] 重构 engine/backtester.py
- [ ] 创建 engine/metrics.py

### 阶段 2：高级分析模块
- [ ] 创建 engine/walk_forward.py
- [ ] 创建 engine/monte_carlo.py
- [ ] 创建 engine/vectorized.py

### 阶段 3：服务层与数据层
- [ ] 优化 services/data_service.py（频率控制）
- [ ] 创建 services/backtest_service.py
- [ ] 更新 services/rate_limiter.py

### 阶段 4：集成与测试
- [ ] 更新 main.py 调用新服务
- [ ] 编写单元测试
- [ ] 端到端测试
- [ ] 性能测试

---

## 9. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| AKShare API 变更 | 保留备用数据源（Sina/NetEase） |
| 数据库性能 | 添加索引，分表策略 |
| 回测速度慢 | 提供向量化回测选项 |
| 参数过拟合 | Walk-Forward 验证 |

---

## 10. 参考资料

- `.agents/skills/backtesting-frameworks/SKILL.md`
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
- [Quantitative Trading (Ernest Chan)](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/1119800064)
