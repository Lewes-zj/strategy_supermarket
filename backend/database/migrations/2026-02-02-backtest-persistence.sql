-- 回测结果入库 - 数据库迁移脚本
-- 创建时间: 2026-02-02

-- 1. 策略交易记录表
CREATE TABLE IF NOT EXISTS strategy_trades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL COMMENT '策略ID',
    trade_date DATE NOT NULL COMMENT '交易日期',
    trade_time VARCHAR(10) COMMENT '交易时间',
    symbol VARCHAR(20) NOT NULL COMMENT '股票代码',
    name VARCHAR(50) COMMENT '股票名称',
    sector VARCHAR(50) COMMENT '行业板块',
    side VARCHAR(10) NOT NULL COMMENT 'buy/sell',
    price FLOAT NOT NULL COMMENT '成交价',
    quantity INT NOT NULL COMMENT '成交数量',
    amount FLOAT NOT NULL COMMENT '成交金额',
    commission FLOAT DEFAULT 0 COMMENT '手续费',
    pnl FLOAT COMMENT '绝对盈亏',
    pnl_pct FLOAT COMMENT '盈亏百分比',
    source VARCHAR(20) DEFAULT 'backtest' COMMENT '数据来源',
    INDEX idx_trade_strategy_date (strategy_id, trade_date),
    INDEX idx_trade_symbol (symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. 策略每日持仓快照表
CREATE TABLE IF NOT EXISTS strategy_daily_snapshots (
    id INT AUTO_INCREMENT PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL COMMENT '策略ID',
    snapshot_date DATE NOT NULL COMMENT '快照日期',
    symbol VARCHAR(20) NOT NULL COMMENT '股票代码',
    name VARCHAR(50) COMMENT '股票名称',
    sector VARCHAR(50) COMMENT '行业板块',
    direction VARCHAR(10) DEFAULT 'Long' COMMENT 'Long/Short',
    quantity INT NOT NULL COMMENT '持仓数量',
    entry_price FLOAT NOT NULL COMMENT '开仓均价',
    current_price FLOAT COMMENT '当日收盘价',
    entry_date DATE NOT NULL COMMENT '开仓日期',
    days_held INT NOT NULL COMMENT '持仓天数',
    weight FLOAT NOT NULL COMMENT '仓位占比',
    floating_pnl FLOAT COMMENT '浮动盈亏金额',
    floating_pnl_pct FLOAT COMMENT '浮动盈亏百分比',
    INDEX idx_snapshot_strategy_date (strategy_id, snapshot_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. 策略每日权益汇总表
CREATE TABLE IF NOT EXISTS strategy_daily_equity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL COMMENT '策略ID',
    equity_date DATE NOT NULL COMMENT '日期',
    total_equity FLOAT NOT NULL COMMENT '总权益',
    cash FLOAT NOT NULL COMMENT '现金',
    position_value FLOAT NOT NULL COMMENT '持仓市值',
    daily_pnl FLOAT COMMENT '当日盈亏',
    daily_pnl_pct FLOAT COMMENT '当日收益率',
    total_pnl FLOAT COMMENT '累计盈亏',
    total_pnl_pct FLOAT COMMENT '累计收益率',
    position_count INT DEFAULT 0 COMMENT '持仓数量',
    INDEX idx_equity_strategy_date (strategy_id, equity_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
