# 策略超市 (Strategy Supermarket) 全栈实施计划

本计划旨在构建“策略超市”的完整全栈应用。前端保持高保真 UI 与营销交互，后端引入真实的 Python 量化系统，负责数据获取、策略回测、信号生成及 API 服务。

## 需要用户审查的内容

> [!IMPORTANT]
> **架构变更：**
> - **前端 (Frontend):** React + TypeScript (Vite) - 负责 UI 展示与营销交互。
> - **后端 (Backend):** Python (FastAPI) - 负责量化逻辑与数据服务。
> - **核心能力:** 集成 `backtesting-frameworks` skill 构建自研回测引擎。
> - **数据源:** 优先接入 `AkShare` (开源) 或模拟数据生成器，确保演示流畅性。

## 详细实施步骤

### 1. 项目初始化 (Project Setup)
- **根目录结构:**
    - `/web`: 前端 React 项目。
    - `/backend`: 后端 Python 项目。
- **环境配置:**
    - Python 3.10+ (Dependencies: `fastapi`, `pandas`, `numpy`, `akshare`, `uvicorn`).
    - Node.js (Vite, React).

### 2. 后端核心：量化回测引擎 (Quant Engine)
基于 `backtesting-frameworks` skill 实现。

#### [新增] `backend/engine/`
- **`backtester.py`**: 实现事件驱动回测框架 (Event-Driven Backtester)。
    - 支持 `Order`, `Fill`, `Position`, `Portfolio` 等核心类。
    - 确保处理交易成本 (Commission) 和滑点 (Slippage)。
- **`data_loader.py`**: 数据加载模块。
    - 封装 `AkShare` 接口获取 A 股/美股日线数据。
    - 实现防止“未来函数” (Look-ahead Bias) 的数据切片逻辑。
- **`strategy_base.py`**: 策略基类。
    - 定义 `on_bar(bar)` 和 `on_fill(fill)` 抽象方法。

#### [新增] `backend/strategies/`
- **`alpha_trend.py`**: 实现 PRD 中的 "Alpha Trend Strategy"。
    - 逻辑示例：双均线突破 + ATR 止损。
    - **实盘模拟:** 每天根据最新数据生成 `Signal`。

### 3. 后端服务：API 接口 (API Layer)
使用 FastAPI 暴露数据给前端。

#### [新增] `backend/main.py` & `backend/api/`
- **`GET /api/strategies`**: 返回策略列表（含计算好的 CAGR, Sharpe 等指标）。
- **`GET /api/strategies/{id}/metrics`**: 返回指定年份的详细回测指标。
- **`GET /api/strategies/{id}/equity_curve`**: 返回净值走势数据 (用于绘制面积图)。
- **`GET /api/strategies/{id}/transactions`**: 返回交易记录。
    - **关键逻辑:** 根据 `is_subscribed` 参数，对“持仓中”的记录进行脱敏处理 (返回 `HIDDEN`)。
- **`GET /api/strategies/{id}/holdings`**: 返回当前持仓。
    - **关键逻辑:** 同样执行严格的脱敏逻辑，仅返回 mock 的“行业”和真实的“浮盈比例”。

### 4. 前端开发 (Frontend Development)
在原计划基础上对接真实 API。

- **调整 `src/mocks/api.ts`**: 替换为真实的 `axios` 或 `fetch` 请求，指向 `http://localhost:8000`。
- **数据展示:**
    - 策略广场：渲染从后端获取的真实的 `CAGR` 和 `Sparkline` 数据。
    - 详情页：图表组件对接后端返回的 `equity_curve` 时间序列。

### 5. 验证与测试 (Verification)

#### 后端验证
- 运行 `backtester.py` 对 "Alpha Trend Strategy" 进行 3 年历史回测，确保无报错。
- 验证回测结果指标 (Sharpe, Drawdown) 计算准确性。
- 验证 `/api/transactions` 接口在未订阅状态下是否正确隐藏了 `symbol` 代码。

#### 前端验证
- 启动 FastAPI 服务和 Vite 开发服务器。
- 访问策略广场，确认卡片加载了后端数据。
- 模拟点击“解锁”，确认前端状态更新后，再次请求 API 能获取到真实代码（模拟订阅状态变化）。

## 下一步行动
1. 初始化 `/backend` 目录。
2. 安装 Python 依赖。
3. 按照 Skill 指引编写 `backtester.py` 核心。
