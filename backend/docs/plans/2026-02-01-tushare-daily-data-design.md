# Tushare日线数据拉取方案设计

## 概述

从tushare拉取A股历史日线数据，支持指定时间范围、增量更新、断点续传等功能。

## 需求

- 数据源：tushare pro（2000积分，500次/分钟）
- 存储：MySQL分区表（按年分区）
- 功能：时间范围参数、增量更新、断点续传、指定股票列表、进度显示

## 数据库设计

### 表结构

```sql
CREATE TABLE tushare_stock_daily (
    ts_code VARCHAR(10) NOT NULL COMMENT 'tushare股票代码(如000001.SZ)',
    trade_date DATE NOT NULL COMMENT '交易日期',
    open DECIMAL(10,2) COMMENT '开盘价',
    high DECIMAL(10,2) COMMENT '最高价',
    low DECIMAL(10,2) COMMENT '最低价',
    close DECIMAL(10,2) COMMENT '收盘价',
    pre_close DECIMAL(10,2) COMMENT '昨收价',
    change_amt DECIMAL(10,2) COMMENT '涨跌额',
    pct_chg DECIMAL(10,4) COMMENT '涨跌幅(%)',
    vol DECIMAL(20,2) COMMENT '成交量(手)',
    amount DECIMAL(20,2) COMMENT '成交额(千元)',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (ts_code, trade_date),
    INDEX idx_trade_date (trade_date),
    INDEX idx_ts_code (ts_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
PARTITION BY RANGE (YEAR(trade_date)) (
    PARTITION p2015 VALUES LESS THAN (2016),
    PARTITION p2016 VALUES LESS THAN (2017),
    PARTITION p2017 VALUES LESS THAN (2018),
    PARTITION p2018 VALUES LESS THAN (2019),
    PARTITION p2019 VALUES LESS THAN (2020),
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p2026 VALUES LESS THAN (2027),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);
```

## 文件结构

```
backend/
├── scripts/
│   └── fetch_tushare_daily.py    # 主脚本入口
├── services/
│   └── tushare_service.py        # tushare数据服务
├── database/
│   └── models.py                  # 新增TushareStockDaily模型
└── data/
    └── .tushare_fetch_progress.json  # 断点续传进度文件
```

## 配置项

```python
# config.py 新增
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
TUSHARE_RATE_LIMIT = float(os.getenv("TUSHARE_RATE_LIMIT", "6.0"))  # 每秒请求数
TUSHARE_RETRY_TIMES = int(os.getenv("TUSHARE_RETRY_TIMES", "3"))    # 重试次数
TUSHARE_RETRY_DELAY = float(os.getenv("TUSHARE_RETRY_DELAY", "1.0")) # 重试间隔(秒)
```

## 命令行接口

```bash
# 基本用法 - 拉取指定时间范围
python scripts/fetch_tushare_daily.py --start 20200101 --end 20241231

# 增量更新 - 只拉取缺失数据
python scripts/fetch_tushare_daily.py --start 20200101 --incremental

# 指定股票列表
python scripts/fetch_tushare_daily.py --start 20200101 --symbols 000001.SZ,600000.SH

# 从文件读取股票列表
python scripts/fetch_tushare_daily.py --start 20200101 --symbols-file stocks.txt

# 继续上次未完成的任务
python scripts/fetch_tushare_daily.py --resume

# 强制重新开始（忽略进度文件）
python scripts/fetch_tushare_daily.py --start 20200101 --force-new
```

## 频率控制

- 2000积分 = 500次/分钟
- 保守策略：6次/秒（360次/分钟），留28%余量
- 批次处理：每100只股票后检查进度

## 断点续传机制

使用单一进度文件 `data/.tushare_fetch_progress.json`：

```json
{
    "task_id": "20241201_103000",
    "start_date": "20200101",
    "end_date": "20241231",
    "total_symbols": 5000,
    "completed_symbols": ["000001.SZ", "000002.SZ"],
    "failed_symbols": {"000003.SZ": "API error: ..."},
    "last_update": "2024-12-01 10:35:00"
}
```

清理机制：
1. 任务正常完成后自动删除进度文件
2. 新任务启动时检测未完成任务，提示用户选择
3. 进度文件超过7天视为过期

## 错误处理

1. API限流错误 → 自动降速并重试
2. 网络超时 → 指数退避重试（1s, 2s, 4s）
3. 股票不存在/退市 → 记录到failed_symbols，继续下一个
4. 数据库写入失败 → 回滚当前批次，记录错误，继续

## 依赖项

```
tushare>=1.2.89
```
