# 龙厂策略 (Dragon Leader Strategy) 设计文档

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个基于连板龙头股的打板策略，在市场情绪好的时候买入连板数最高的龙头股。

**Architecture:** 使用 AKShare 的涨停股池接口获取连板数据，预加载回测期间所有涨停股的行情数据，通过事件驱动回测引擎执行策略逻辑。

**Tech Stack:** Python, AKShare (stock_zt_pool_em, stock_zt_pool_dtgc_em), SQLAlchemy, pandas

---

## 策略核心逻辑

### 选股逻辑 (T日盘前，使用T-1日数据)
1. 从涨停股池获取昨日(T-1)所有涨停股票及其连板数
2. 筛选出连板数最高的股票（最高板）
3. 排除条件：
   - ST/退市股票
   - 科创板(688)股票
   - 7天内已买卖过的股票
4. 额外过滤：MA60 上升趋势（可选）

### 买入条件 (T日盘中，11:00前)
1. 大盘趋势向上（上证指数近3天线性拟合斜率 > 0）
2. 连板数 ≥ 3
3. 非一字涨停连板数 ≥ 2（通过炸板次数 > 0 或 首次封板时间 > 09:30 判断）
4. 当前未涨停（避免追高）
5. 最多持有1只股票

### 卖出条件
1. 尾盘(14:55后)未涨停 → 卖出
2. 亏损超过5% → 止损卖出

### 风险管理
如果昨日跌停数 > 前日跌停数 且 昨日跌停数 > 15，当日暂停买入

---

## 数据流设计

### 预加载流程
```
回测启动时:
├── Step 1: 获取交易日历 (从 stock_daily 表提取)
├── Step 2: 预加载涨停/跌停数据 (缓存到内存)
├── Step 3: 提取所有候选股票代码 (去重)
├── Step 4: 增量加载股票行情数据 (检查本地，只下载缺失)
└── Step 5: 开始回测
```

### 增量加载逻辑
- 先查询本地 stock_daily 表已有数据的日期范围
- 只下载缺失的部分，避免重复请求 AKShare
- 第二次回测秒开

---

## 文件结构

```
strategies/
└── dragon_strategy.py    # 龙厂策略

services/
└── zt_pool_service.py    # 涨停股池数据服务
```

---

## 策略参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_hold_num` | 1 | 最多持有股票数 |
| `min_board_count` | 3 | 最低连板数要求 |
| `min_non_yz_board` | 2 | 非一字板最低连板数 |
| `stop_loss_pct` | 0.05 | 止损比例 (5%) |
| `market_risk_threshold` | 15 | 跌停数超过此值触发风控 |
| `cooldown_days` | 7 | 同一股票买卖冷却期 |
| `buy_before_hour` | 11 | 只在此时间前买入 |

---

## AKShare 接口

| 接口 | 用途 | 关键字段 |
|------|------|----------|
| `stock_zt_pool_em(date)` | 涨停股池 | 代码, 名称, 连板数, 炸板次数, 首次封板时间 |
| `stock_zt_pool_dtgc_em(date)` | 跌停股池 | 代码, 连续跌停 |

**重要**：盘前选股时必须传入 T-1 日的日期，不是当日。
