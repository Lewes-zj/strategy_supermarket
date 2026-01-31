# 年份切换功能 Bug 修复设计文档

**日期**: 2026-01-29
**状态**: 待实施
**影响范围**: 后端 API + 前端组件 + 缓存层

---

## 1. 问题描述

策略详情页左侧年份切换后，右侧数据（Performance Snapshot、累积收益图、交易记录）不会更新为对应年份的数据。

## 2. 根因分析

### 2.1 前端层问题

| 文件 | 问题 |
|------|------|
| `DataMetrics.tsx:21-22` | 调用 API 时没有传递 `selectedYear` 参数 |
| `TransactionHistory.tsx` | 完全没有使用 `selectedYear` |
| `api.ts:getEquityCurve()` | 不支持 `year` 参数 |

### 2.2 后端层问题

| 端点 | 问题 |
|------|------|
| `/api/strategies/{id}/metrics` | 有 `year` 参数但未实现筛选逻辑 |
| `/api/strategies/{id}/equity_curve` | 无 `year` 参数 |
| `/api/strategies/{id}/drawdown` | 无 `year` 参数 |
| `/api/strategies/{id}/monthly-returns` | 无 `year` 参数 |
| `/api/strategies/{id}/transactions` | 无 `year` 参数 |

### 2.3 当前缓存机制

- MySQL 数据库缓存（StrategyBacktest 表）：24 小时 TTL
- 内存字典缓存（STRATEGY_RESULTS）：1 分钟 TTL
- 缓存是全周期数据，没有按年份分开

---

## 3. 解决方案

采用 **后端筛选 + Redis 缓存** 方案。

### 3.1 后端 API 改造

#### 需要修改的端点

```python
# 1. /api/strategies/{id}/metrics - 实现年份筛选
@app.get("/api/strategies/{id}/metrics")
def get_metrics(id: str, year: Optional[int] = None):
    res = run_backtest(id, use_real_data=True)
    df = res["equity"]

    if year:
        df = df[df.index.year == year]
        metrics = _calculate_metrics(df["returns"])
    else:
        metrics = res["metrics"]

    return metrics

# 2. /api/strategies/{id}/equity_curve - 添加 year 参数
@app.get("/api/strategies/{id}/equity_curve")
def get_equity_curve(id: str, year: Optional[int] = None):
    res = run_backtest(id, use_real_data=True)
    df = res["equity"]

    if year:
        df = df[df.index.year == year]

    return [{"date": ts.strftime("%Y-%m-%d"), "value": float(val), ...} for ...]

# 3. /api/strategies/{id}/drawdown - 添加 year 参数
@app.get("/api/strategies/{id}/drawdown")
def get_drawdown(id: str, year: Optional[int] = None):
    # 类似逻辑

# 4. /api/strategies/{id}/monthly-returns - 添加 year 参数
@app.get("/api/strategies/{id}/monthly-returns")
def get_monthly_returns(id: str, year: Optional[int] = None):
    # 类似逻辑

# 5. /api/strategies/{id}/transactions - 添加 year 参数
@app.get("/api/strategies/{id}/transactions")
def get_transactions(id: str, is_subscribed: bool = False, year: Optional[int] = None):
    # 类似逻辑
```

#### Holdings 端点特殊处理

`/api/strategies/{id}/holdings` 显示当前实时持仓，不需要按年份筛选。

### 3.2 Redis 缓存设计

#### 缓存架构

```
前端请求 ──▶ Redis 缓存 ──miss──▶ MySQL 计算 ──▶ 写入 Redis
                │
                ▼ hit
             返回数据
```

#### 缓存 Key 设计

| 数据类型 | Key 格式 | 示例 |
|---------|---------|------|
| 年份指标 | `strategy:{id}:metrics:{year}` | `strategy:alpha_trend:metrics:2024` |
| 净值曲线 | `strategy:{id}:equity:{year}` | `strategy:alpha_trend:equity:2024` |
| 回撤数据 | `strategy:{id}:drawdown:{year}` | `strategy:alpha_trend:drawdown:2024` |
| 月度收益 | `strategy:{id}:monthly:{year}` | `strategy:alpha_trend:monthly:2024` |
| 交易记录 | `strategy:{id}:transactions:{year}` | `strategy:alpha_trend:transactions:2024` |
| 全周期数据 | `strategy:{id}:metrics:all` | `strategy:alpha_trend:metrics:all` |

#### TTL 策略

| 数据类型 | TTL | 原因 |
|---------|-----|------|
| 历史年份数据 | 24 小时 | 数据不变，可以长缓存 |
| 当前年份数据 | 1 小时 | 可能有新交易 |
| 全周期汇总 | 1 小时 | 随当前年份变化 |

#### 缓存失效机制

```python
# 当触发新回测或数据更新时
def invalidate_strategy_cache(strategy_id: str):
    redis.delete_pattern(f"strategy:{strategy_id}:*")
```

### 3.3 前端改造

#### api.ts 修改

```typescript
// 添加/修改 year 参数
getMetrics: async (id: string, year?: number): Promise<Metrics> => {
    const res = await axios.get(`${API_BASE}/strategies/${id}/metrics`, {
        params: year ? { year } : undefined
    });
    return res.data;
},

getEquityCurve: async (id: string, year?: number): Promise<EquityPoint[]> => {
    const res = await axios.get(`${API_BASE}/strategies/${id}/equity_curve`, {
        params: year ? { year } : undefined
    });
    return res.data;
},

getDrawdown: async (id: string, year?: number): Promise<DrawdownPoint[]> => {
    const res = await axios.get(`${API_BASE}/strategies/${id}/drawdown`, {
        params: year ? { year } : undefined
    });
    return res.data;
},

getMonthlyReturns: async (id: string, year?: number): Promise<MonthlyReturn[]> => {
    const res = await axios.get(`${API_BASE}/strategies/${id}/monthly-returns`, {
        params: year ? { year } : undefined
    });
    return res.data;
},

getTransactions: async (id: string, isSubscribed: boolean, year?: number): Promise<Transaction[]> => {
    const res = await axios.get(`${API_BASE}/strategies/${id}/transactions`, {
        params: { is_subscribed: isSubscribed, ...(year ? { year } : {}) }
    });
    return res.data;
},
```

#### DataMetrics.tsx 修改

```typescript
useEffect(() => {
    const fetchData = async () => {
        if (id) {
            try {
                // 传递 selectedYear 参数
                const m = await api.getMetrics(id, selectedYear);
                const e = await api.getEquityCurve(id, selectedYear);
                const d = await api.getDrawdown(id, selectedYear);
                const monthly = await api.getMonthlyReturns(id, selectedYear);

                setMetrics(m);
                setEquityData(e);
                setDrawdownData(d);
                setMonthlyData(monthly);
            } catch (err) {
                console.error(err);
            }
        }
    };
    fetchData();
}, [id, selectedYear]);
```

#### TransactionHistory.tsx 修改

```typescript
// 从 context 获取 selectedYear
const { isSubscribed, requestSubscribe, selectedYear } = useOutletContext<{
    isSubscribed: boolean;
    requestSubscribe: () => void;
    selectedYear: number;
}>();

useEffect(() => {
    const fetchData = async () => {
        if (id) {
            try {
                // 传递 selectedYear 参数
                const data = await api.getTransactions(id, isSubscribed, selectedYear);
                setTransactions(data);
            } catch (e) {
                console.error(e);
            }
        }
    };
    fetchData();
}, [id, isSubscribed, selectedYear]);
```

---

## 4. 实施步骤

### 阶段 1：后端基础改造

1. 安装 Redis 依赖
   ```bash
   pip install redis
   ```

2. 创建 `backend/services/cache_service.py`
   - Redis 连接管理
   - 缓存读写封装
   - Key 生成工具函数

3. 修改 `backend/config.py`
   - 添加 Redis 配置项

4. 修改 `backend/main.py`
   - 5 个 API 端点添加年份筛选逻辑

### 阶段 2：缓存集成

5. 在各端点中集成 Redis 缓存
   - 先查缓存
   - 缓存未命中时计算并写入
   - 返回数据

6. 添加缓存失效机制
   - 数据更新时清除相关缓存

### 阶段 3：前端适配

7. 修改 `web/src/services/api.ts`
   - 添加/修改 year 参数

8. 修改 `web/src/pages/tabs/DataMetrics.tsx`
   - 传递 selectedYear 到 API 调用

9. 修改 `web/src/pages/tabs/TransactionHistory.tsx`
   - 从 context 获取 selectedYear
   - 传递到 API 调用

### 阶段 4：测试验证

10. 功能测试
    - 年份切换后数据正确更新
    - 各 Tab 页数据一致性

11. 缓存测试
    - 缓存命中率验证
    - 缓存失效正确性

12. 边界情况测试
    - 无数据年份处理
    - 当前年份（RUNNING）处理
    - 全周期（Total）选项处理

---

## 5. 文件清单

### 需要修改的文件

| 文件路径 | 修改内容 |
|---------|---------|
| `backend/main.py` | 5 个 API 端点添加年份筛选 |
| `backend/config.py` | 添加 Redis 配置 |
| `web/src/services/api.ts` | 添加 year 参数 |
| `web/src/pages/tabs/DataMetrics.tsx` | 传递 selectedYear |
| `web/src/pages/tabs/TransactionHistory.tsx` | 传递 selectedYear |

### 需要新增的文件

| 文件路径 | 内容 |
|---------|------|
| `backend/services/cache_service.py` | Redis 缓存服务 |

---

## 6. 风险与注意事项

1. **Redis 依赖**: 需确保生产环境 Redis 服务可用，建议添加降级逻辑（Redis 不可用时直接查库）

2. **缓存一致性**: 数据更新后需及时清除缓存，避免脏数据

3. **空数据处理**: 某些年份可能无数据，前端需优雅处理

4. **向后兼容**: year 参数为可选，不传时返回全周期数据，保持现有行为
