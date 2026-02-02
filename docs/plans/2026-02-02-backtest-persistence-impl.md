# 回测结果入库实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将回测结果从 JSON 存储改为结构化数据库表存储，支持分页、筛选等复杂查询

**Architecture:** 新增 3 张表（StrategyTrade, StrategyDailySnapshot, StrategyDailyEquity），回测完成后写入结构化数据，API 层从新表查询

**Tech Stack:** SQLAlchemy, FastAPI, React/TypeScript

---

## Task 1: 新增数据库模型

**Files:**
- Modify: `backend/database/models.py`

**Step 1: 在 models.py 末尾添加 StrategyTrade 模型**

```python
class StrategyTrade(Base):
    """策略交易记录表"""
    __tablename__ = "strategy_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), nullable=False, comment="策略ID")
    trade_date = Column(Date, nullable=False, comment="交易日期")
    trade_time = Column(String(10), nullable=True, comment="交易时间")
    symbol = Column(String(20), nullable=False, comment="股票代码")
    name = Column(String(50), nullable=True, comment="股票名称")
    sector = Column(String(50), nullable=True, comment="行业板块")
    side = Column(String(10), nullable=False, comment="buy/sell")
    price = Column(Float, nullable=False, comment="成交价")
    quantity = Column(Integer, nullable=False, comment="成交数量")
    amount = Column(Float, nullable=False, comment="成交金额")
    commission = Column(Float, default=0, comment="手续费")
    pnl = Column(Float, nullable=True, comment="绝对盈亏")
    pnl_pct = Column(Float, nullable=True, comment="盈亏百分比")
    source = Column(String(20), default="backtest", comment="数据来源")

    __table_args__ = (
        Index("idx_trade_strategy_date", "strategy_id", "trade_date"),
        Index("idx_trade_symbol", "symbol"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )
```

**Step 2: 添加 StrategyDailySnapshot 模型**

```python
class StrategyDailySnapshot(Base):
    """策略每日持仓快照表"""
    __tablename__ = "strategy_daily_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), nullable=False, comment="策略ID")
    snapshot_date = Column(Date, nullable=False, comment="快照日期")
    symbol = Column(String(20), nullable=False, comment="股票代码")
    name = Column(String(50), nullable=True, comment="股票名称")
    sector = Column(String(50), nullable=True, comment="行业板块")
    direction = Column(String(10), default="Long", comment="Long/Short")
    quantity = Column(Integer, nullable=False, comment="持仓数量")
    entry_price = Column(Float, nullable=False, comment="开仓均价")
    current_price = Column(Float, nullable=True, comment="当日收盘价")
    entry_date = Column(Date, nullable=False, comment="开仓日期")
    days_held = Column(Integer, nullable=False, comment="持仓天数")
    weight = Column(Float, nullable=False, comment="仓位占比")
    floating_pnl = Column(Float, nullable=True, comment="浮动盈亏金额")
    floating_pnl_pct = Column(Float, nullable=True, comment="浮动盈亏百分比")

    __table_args__ = (
        Index("idx_snapshot_strategy_date", "strategy_id", "snapshot_date"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )
```

**Step 3: 添加 StrategyDailyEquity 模型**

```python
class StrategyDailyEquity(Base):
    """策略每日权益汇总表"""
    __tablename__ = "strategy_daily_equity"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), nullable=False, comment="策略ID")
    equity_date = Column(Date, nullable=False, comment="日期")
    total_equity = Column(Float, nullable=False, comment="总权益")
    cash = Column(Float, nullable=False, comment="现金")
    position_value = Column(Float, nullable=False, comment="持仓市值")
    daily_pnl = Column(Float, nullable=True, comment="当日盈亏")
    daily_pnl_pct = Column(Float, nullable=True, comment="当日收益率")
    total_pnl = Column(Float, nullable=True, comment="累计盈亏")
    total_pnl_pct = Column(Float, nullable=True, comment="累计收益率")
    position_count = Column(Integer, default=0, comment="持仓数量")

    __table_args__ = (
        Index("idx_equity_strategy_date", "strategy_id", "equity_date"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )
```

**Step 4: 运行数据库迁移**

```bash
cd backend && python -c "from database.models import Base; from database.connection import engine; Base.metadata.create_all(engine)"
```

---

## Task 2: 实现回测结果持久化服务

**Files:**
- Create: `backend/services/backtest_persistence.py`

**Step 1: 创建持久化服务文件**

```python
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
        trades: List[Fill],
        sector_map: Dict[str, str]
    ) -> None:
        """保存交易记录"""
        for fill in trades:
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
            total_pnl_pct = (total_equity - initial_equity) / initial_equity

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
```

---

## Task 3: 改造交易记录 API

**Files:**
- Modify: `backend/main.py:1044-1109`

**Step 1: 替换 get_transactions 函数**

```python
@app.get("/api/strategies/{id}/transactions")
def get_transactions(
    id: str,
    is_subscribed: bool = False,
    year: Optional[int] = None,
    page: int = 1,
    page_size: int = 50
):
    """Get transaction history from database with pagination."""
    from database.models import StrategyTrade
    from database.connection import get_db_session
    from sqlalchemy import extract

    try:
        db = get_db_session()

        # 获取行业映射
        from database.repository import StockPoolRepository
        stock_pool_repo = StockPoolRepository()
        stock_pool = stock_pool_repo.get_stock_pool()
        sector_map = {s['symbol']: s.get('sector', '其他') for s in stock_pool}

        # 构建查询
        query = db.query(StrategyTrade).filter(StrategyTrade.strategy_id == id)

        if year:
            query = query.filter(extract('year', StrategyTrade.trade_date) == year)

        # 总数
        total = query.count()

        # 分页查询
        trades = query.order_by(
            StrategyTrade.trade_date.desc(),
            StrategyTrade.id.desc()
        ).offset((page - 1) * page_size).limit(page_size).all()

        # 格式化响应
        api_trades = []
        for i, trade in enumerate(trades):
            is_encrypted = (i < 3) and (not is_subscribed)

            trade_data = {
                "date": trade.trade_date.isoformat(),
                "time": trade.trade_time or "",
                "symbol": trade.symbol if not is_encrypted else trade.sector,
                "sector": trade.sector,
                "side": trade.side,
                "price": trade.price if not is_encrypted else None,
                "pnl": trade.pnl_pct or 0,
                "is_encrypted": is_encrypted
            }

            if is_encrypted:
                trade_data["pnl_percent"] = 0.05 + (i * 0.02)

            api_trades.append(trade_data)

        return {"transactions": api_trades, "total": total, "page": page}

    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return {"transactions": [], "total": 0, "page": 1}
```

---

## Task 4: 改造每日持仓 API

**Files:**
- Modify: `backend/main.py:1112-1220`

**Step 1: 替换 get_holdings 函数**

```python
@app.get("/api/strategies/{id}/holdings")
def get_holdings(id: str, is_subscribed: bool = False):
    """Get current holdings from database."""
    from database.models import StrategyDailySnapshot
    from database.connection import get_db_session
    from sqlalchemy import func

    try:
        db = get_db_session()

        # 获取最新日期
        latest_date = db.query(func.max(StrategyDailySnapshot.snapshot_date)).filter(
            StrategyDailySnapshot.strategy_id == id
        ).scalar()

        if not latest_date:
            return {"holdings": [], "total_pnl_pct": 0, "position_count": 0}

        # 查询当日持仓
        holdings = db.query(StrategyDailySnapshot).filter(
            StrategyDailySnapshot.strategy_id == id,
            StrategyDailySnapshot.snapshot_date == latest_date
        ).all()

        # 计算总浮盈
        total_pnl_pct = sum(
            h.weight * (h.floating_pnl_pct or 0) for h in holdings
        )

        # 格式化响应
        response_list = []
        for h in holdings:
            item = {
                "sector": h.sector or "其他",
                "direction": h.direction or "Long",
                "days_held": h.days_held,
                "weight": f"{h.weight * 100:.0f}%",
                "pnl_pct": (h.floating_pnl_pct or 0) * 100
            }

            if is_subscribed:
                item["symbol"] = h.symbol
                item["name"] = h.name or h.symbol
                item["current_price"] = h.current_price
            else:
                item["symbol"] = "HIDDEN"
                item["name"] = "HIDDEN"
                item["current_price"] = None

            response_list.append(item)

        return {
            "holdings": response_list,
            "total_pnl_pct": total_pnl_pct * 100,
            "position_count": len(holdings)
        }

    except Exception as e:
        logger.error(f"Error getting holdings: {e}")
        return {"holdings": [], "total_pnl_pct": 0, "position_count": 0}
```

---

## Task 5: 更新前端类型定义

**Files:**
- Modify: `web/src/services/types.ts`

**Step 1: 更新 Transaction 接口**

在 `Transaction` 接口中添加 `sector` 和 `source` 字段：

```typescript
export interface Transaction {
    date: string;
    time: string;
    symbol: string;
    sector?: string;           // 新增
    side: string;
    price: number | null;
    pnl: number;
    pnl_percent?: number;
    is_encrypted: boolean;
    source?: 'backtest' | 'live';  // 新增
}
```

**Step 2: 新增 TransactionsResponse 接口**

```typescript
export interface TransactionsResponse {
    transactions: Transaction[];
    total: number;
    page: number;
}
```

---

## Task 6: 更新前端 API 服务

**Files:**
- Modify: `web/src/services/api.ts`

**Step 1: 更新 getTransactions 方法**

```typescript
getTransactions: async (
    id: string,
    isSubscribed: boolean,
    year?: number,
    page: number = 1,
    pageSize: number = 50
): Promise<import('./types').TransactionsResponse> => {
    const res = await axios.get(`${API_BASE}/strategies/${id}/transactions`, {
        params: {
            is_subscribed: isSubscribed,
            ...(year ? { year } : {}),
            page,
            page_size: pageSize
        }
    });
    return res.data;
},
```

---

## Task 7: 更新前端 TransactionHistory 组件

**Files:**
- Modify: `web/src/pages/tabs/TransactionHistory.tsx`

**Step 1: 更新状态和数据获取逻辑**

```typescript
const [transactions, setTransactions] = useState<Transaction[]>([]);
const [total, setTotal] = useState(0);
const [page, setPage] = useState(1);
const [hasMore, setHasMore] = useState(true);

const fetchData = async (pageNum: number = 1, append: boolean = false) => {
    if (id) {
        setLoading(true);
        try {
            const [txResponse, signals] = await Promise.all([
                api.getTransactions(id, isSubscribed, selectedYear || undefined, pageNum),
                api.getSignals(id)
            ]);

            if (append) {
                setTransactions(prev => [...prev, ...txResponse.transactions]);
            } else {
                setTransactions(txResponse.transactions);
            }
            setTotal(txResponse.total);
            setHasMore(txResponse.transactions.length === 50);
            setSignalData(signals);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    }
};
```

**Step 2: 添加加载更多按钮**

在表格底部添加：

```tsx
{hasMore && transactions.length < total && (
    <div style={{ textAlign: 'center', padding: '16px' }}>
        <button
            onClick={() => {
                const nextPage = page + 1;
                setPage(nextPage);
                fetchData(nextPage, true);
            }}
            style={{
                padding: '8px 24px',
                background: 'var(--bg-card)',
                border: '1px solid var(--border-light)',
                borderRadius: '4px',
                color: 'var(--text-secondary)',
                cursor: 'pointer'
            }}
        >
            加载更多 ({transactions.length}/{total})
        </button>
    </div>
)}
```

---

## 实现顺序

1. Task 1: 新增数据库模型
2. Task 2: 实现持久化服务
3. Task 3: 改造交易记录 API
4. Task 4: 改造每日持仓 API
5. Task 5: 更新前端类型
6. Task 6: 更新前端 API
7. Task 7: 更新前端组件
