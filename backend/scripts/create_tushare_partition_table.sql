-- 创建tushare日线数据分区表
-- 注意：SQLAlchemy不支持自动创建分区表，需要手动执行此SQL
-- 用法: mysql -u root -p strategy_market < backend/scripts/create_tushare_partition_table.sql

DROP TABLE IF EXISTS tushare_stock_daily;

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
