# 回测结果入库设计方案

| 版本 | 日期 | 修改人 | 备注 |
|------|------|--------|------|
| v1.0 | 2026-02-02 | Claude | 初始设计 |

## 1. 背景与目标

### 1.1 核心问题

当前回测结果以 JSON 形式存储在 `StrategyBacktest` 表中，存在以下问题：

1. **性能问题** - JSON 存储查询慢，无法支持按日期筛选、分页等
2. **数据完整性** - 缓存机制不够可靠，需要更持久化的存储方案
3. **功能扩展** - 无法支持复杂查询（如按年份筛选交易记录、持仓历史追溯等）

### 1.2 设计目标

- 将回测结果从 JSON 存储改为结构化数据库表存储
- 支持回测数据和实盘数据混合存储
- 支持增量更新，历史数据不变
- 前端支持分页、筛选等复杂查询

---

## 2. 数据库表结构

### 2.1 StrategyTrade 表 - 交易记录

```python
class StrategyTrade(Base):
    """策略交易记录表"""
    __tablename__ = "strategy_trades"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(50), index=True, nullable=False)

    # 交易基本信息
    trade_date = Column(Date, index=True, nullable=False)      # 交易日期
    trade_time = Column(Time, nullable=True)                   # 交易时间
    symbol = Column(String(20), index=True, nullable=False)    # 股票代码
    name = Column(String(50), nullable=True)                   # 股票名称
    sector = Column(String(50), nullable=True)                 # 行业板块

    # 交易详情
    side = Column(String(10), nullable=False)                  # buy/sell
    price = Column(Numeric(10, 2), nullable=False)             # 成交价
    quantity = Column(Integer, nullable=False)                 # 成交数量
    amount = Column(Numeric(15, 2), nullable=False)            # 成交金额
    commission = Column(Numeric(10, 2), default=0)             # 手续费

    # 盈亏（仅卖出时有值）
    pnl = Column(Numeric(15, 2), nullable=True)                # 绝对盈亏
    pnl_pct = Column(Numeric(8, 4), nullable=True)             # 盈亏百分比

    # 数据来源
    source = Column(String(20), default="backtest")            # backtest/live

    __table_args__ = (
        Index('ix_strategy_trade_lookup', 'strategy_id', 'trade_date'),
    )
```

### 2.2 StrategyDailySnapshot 表 - 每日持仓快照

```python
class StrategyDailySnapshot(Base):
    """策略每日持仓快照表 - 仅记录个股持仓"""
    __tablename__ = "strategy_daily_snapshots"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(50), index=True, nullable=False)
    snapshot_date = Column(Date, index=True, nullable=False)

    # 持仓信息
    symbol = Column(String(20), nullable=False)
    name = Column(String(50), nullable=True)
    sector = Column(String(50), nullable=True)
    direction = Column(String(10), default="Long")             # Long/Short

    # 持仓详情
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Numeric(10, 2), nullable=False)
    current_price = Column(Numeric(10, 2), nullable=True)
    entry_date = Column(Date, nullable=False)
    days_held = Column(Integer, nullable=False)

    # 仓位与盈亏
    weight = Column(Numeric(6, 4), nullable=False)
    floating_pnl = Column(Numeric(15, 2), nullable=True)
    floating_pnl_pct = Column(Numeric(8, 4), nullable=True)

    __table_args__ = (
        UniqueConstraint('strategy_id', 'snapshot_date', 'symbol', name='uq_daily_snapshot'),
    )
```

### 2.3 StrategyDailyEquity 表 - 每日权益汇总

```python
class StrategyDailyEquity(Base):
    """策略每日权益汇总表 - 组合级别数据"""
    __tablename__ = "strategy_daily_equity"

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(50), index=True, nullable=False)
    equity_date = Column(Date, index=True, nullable=False)

    # 组合级别数据
    total_equity = Column(Numeric(15, 2), nullable=False)      # 总权益
    cash = Column(Numeric(15, 2), nullable=False)              # 现金
    position_value = Column(Numeric(15, 2), nullable=False)    # 持仓市值

    # 盈亏
    daily_pnl = Column(Numeric(15, 2), nullable=True)          # 当日盈亏
    daily_pnl_pct = Column(Numeric(8, 4), nullable=True)       # 当日收益率
    total_pnl = Column(Numeric(15, 2), nullable=True)          # 累计盈亏
    total_pnl_pct = Column(Numeric(8, 4), nullable=True)       # 累计收益率

    # 持仓统计
    position_count = Column(Integer, default=0)                # 持仓数量

    __table_args__ = (
        UniqueConstraint('strategy_id', 'equity_date', name='uq_daily_equity'),
    )
```

---

## 3. 数据写入流程

### 3.1 回测完成后的数据写入

```
回测完成 (BacktestResult)
    │
    ├─► 遍历 trades 列表
    │       └─► 写入 StrategyTrade 表
    │
    ├─► 遍历 equity_curve 的每一天
    │       ├─► 写入 StrategyDailyEquity 表
    │       └─► 根据当日持仓写入 StrategyDailySnapshot 表
    │
    └─► 更新 StrategyBacktest 表的 metrics 字段
```

### 3.2 实盘信号触发后的数据写入

```
实盘信号触发 (StrategySignal)
    │
    ├─► 写入 StrategyTrade 表 (source='live')
    │
    └─► 更新当日 StrategyDailySnapshot
```

### 3.3 数据来源区分

- 回测数据和实盘数据通过 `source` 字段区分
- 两种数据共存，查询时可按 source 筛选
- 不会因为实盘数据而丢失回测历史

---

## 4. API 层改造

### 4.1 交易记录 API

**端点**: `GET /api/strategies/{id}/transactions`

**参数**:
- `is_subscribed`: 是否订阅
- `year`: 年份筛选
- `page`: 页码
- `page_size`: 每页数量

**响应**:
```json
{
    "transactions": [...],
    "total": 100,
    "page": 1
}
```

### 4.2 每日持仓 API

**端点**: `GET /api/strategies/{id}/holdings`

**改造**: 从 `StrategyDailySnapshot` 表查询最新日期的持仓

### 4.3 权益曲线 API

**端点**: `GET /api/strategies/{id}/equity_curve`

**改造**: 从 `StrategyDailyEquity` 表查询

---

## 5. 前端改造

### 5.1 类型定义更新

```typescript
export interface TransactionsResponse {
    transactions: Transaction[];
    total: number;
    page: number;
}
```

### 5.2 API 服务更新

- `getTransactions` 支持分页参数
- `getEquityCurve` 返回结构化数据

### 5.3 组件更新

- `TransactionHistory` 支持"加载更多"交互

---

## 6. 实现计划

### 6.1 需要修改的文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `backend/database/models.py` | 修改 | 新增 3 张表定义 |
| `backend/engine/backtester.py` | 修改 | 回测完成后调用数据写入 |
| `backend/services/backtest_service.py` | 修改 | 新增 `save_backtest_result` 方法 |
| `backend/main.py` | 修改 | 改造 3 个 API 端点 |
| `web/src/services/types.ts` | 修改 | 更新类型定义 |
| `web/src/services/api.ts` | 修改 | 更新 API 调用 |
| `web/src/pages/tabs/TransactionHistory.tsx` | 修改 | 支持分页 |

### 6.2 实现步骤

1. **数据库层**: 新增 3 张表定义，运行迁移
2. **服务层**: 实现 `save_backtest_result` 方法
3. **回测器**: 回测完成后调用持久化
4. **API 层**: 改造 3 个端点从新表查询
5. **前端**: 更新类型和组件
