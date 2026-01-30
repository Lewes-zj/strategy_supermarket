# Strategy Supermarket - Backend

量化策略超���后端服务，基于 FastAPI + MySQL + AkShare 实现。

## 功能特性

- ✅ **策略回测引擎**: 事件驱动的回测框架，支持自定义策略
- ✅ **AkShare集成**: 自动获取A股实时数据，带请求频率控制
- ✅ **多策略支持**: Alpha趋势、均值回归、动量、板块轮动
- ✅ **沪深300股票池**: 自动��步沪深300成分股
- ✅ **定时任务**: 自动更新数据、生成交易信号
- ✅ **实时信号**: 策略信号实时推送

## 环境要求

- Python 3.9+
- MySQL 5.7+ 或 8.0+

## 安装步骤

### 1. 创建虚拟环境

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置MySQL

创建数据库:
```sql
CREATE DATABASE strategy_market CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

复制配置文件:
```bash
cp .env.example .env
```

编辑 `.env` 文件，配置MySQL连接信息:
```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=strategy_market
```

### 4. 初始化数据

```bash
# 初始化股票池和获取历史数据
python scripts/init_stock_pool.py
```

## 运行服务

```bash
python main.py
```

服务将在 http://localhost:8000 启动

## API端点

| 端点 | 说明 |
|------|------|
| `GET /api/strategies` | 获取策略列表（支持搜索、排序） |
| `GET /api/strategies/{id}/metrics` | 获取策略指标 |
| `GET /api/strategies/{id}/equity_curve` | 获取权益曲线 |
| `GET /api/strategies/{id}/transactions` | 获取交易记录 |
| `GET /api/strategies/{id}/holdings` | 获取当前持仓 |
| `GET /api/strategies/{id}/signals` | 获取实时信号 |
| `GET /api/market/symbols` | 获取股票列表 |
| `GET /api/market/sectors` | 获取行业板块 |
| `POST /api/admin/update-data` | 手动触发数据更新 |

## 项目结构

```
backend/
├── main.py                  # FastAPI主应用
├── config.py                # 配置管理
├── requirements.txt         # Python依赖
├── .env.example             # 环境变量模板
├── database/                # 数据库模块
│   ├── models.py           # SQLAlchemy模型
│   ├── connection.py       # 数据库连接
│   └── repository.py       # 数据访问层
├── engine/                  # 回测引擎
│   ├── backtester.py       # 核心回测框架
│   └── data_loader.py      # 数据加载器
├── strategies/              # 策略实现
│   ├── registry.py         # 策略注册表
│   ├── alpha_trend.py      # Alpha趋势策略
│   ├── mean_reversion.py   # 均值回归策略
│   ├── momentum.py         # 动量策略
│   └── sector_rotation.py  # 板块轮动策略
├── services/                # 服务层
│   ├── data_service.py     # 数据服务
│   └── signal_service.py   # 信号服务
├── utils/                   # 工具类
│   └── rate_limiter.py     # 请求节流器
├── scheduler.py             # 定时任务
└── scripts/                 # 脚本
    └── init_stock_pool.py   # 初始化脚本
```

## 开发说明

### 添加新策略

1. 在 `strategies/` 目录创建新文件，继承 `Strategy` 类
2. 在 `strategies/registry.py` 中注册策略
3. 重启服务即可自动生效

### 调整AkShare频率

在 `.env` 文件中修改:
```
AKSHARE_RATE_LIMIT=1.0  # 每秒请求数
AKSHARE_BURST_SIZE=10    # 突发容量
```

## 注意事项

1. **AkShare频率**: 默认每秒1次请求，过于频繁会被封IP
2. **数据实时性**: 日线数据收盘后更新，实时行情节流更新
3. **MySQL连接**: 确保MySQL服务正在运行，且配置正确
