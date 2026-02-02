"""
SQLAlchemy database models for Strategy Supermarket.
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, Date, Index
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class StockDaily(Base):
    """Daily stock price data (OHLCV)."""
    __tablename__ = "stock_daily"

    symbol = Column(String(10), primary_key=True, comment="股票代码")
    trade_date = Column(Date, primary_key=True, comment="交易日期")
    open = Column(Float, nullable=False, comment="开盘价")
    high = Column(Float, nullable=False, comment="最高价")
    low = Column(Float, nullable=False, comment="最低价")
    close = Column(Float, nullable=False, comment="收盘价")
    volume = Column(Integer, nullable=False, comment="成交量")
    amount = Column(Float, comment="成交额")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    __table_args__ = (
        Index("idx_symbol_date", "symbol", "trade_date"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class StrategyBacktest(Base):
    """Cached backtest results for strategies."""
    __tablename__ = "strategy_backtest"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), nullable=False, comment="策略ID")
    symbols = Column(String(500), comment="交易股票列表(JSON)")
    start_date = Column(Date, nullable=False, comment="回测开始日期")
    end_date = Column(Date, nullable=False, comment="回测结束日期")
    equity_curve = Column(LONGTEXT, comment="权益曲线数据(JSON)")
    metrics = Column(Text, nullable=False, comment="策略指标(JSON)")
    trades = Column(Text, comment="交易记录(JSON)")
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="缓存时间")

    __table_args__ = (
        Index("idx_strategy_date", "strategy_id", "start_date", "end_date"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class StrategySignal(Base):
    """Real-time trading signals from strategies."""
    __tablename__ = "strategy_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), nullable=False, comment="策略ID")
    symbol = Column(String(10), nullable=False, comment="股票代码")
    signal_type = Column(String(10), nullable=False, comment="信号类型(buy/sell)")
    price = Column(Float, comment="信号价格")
    quantity = Column(Integer, comment="建议数量")
    reason = Column(String(200), comment="信号原因")
    is_active = Column(Boolean, default=True, comment="是否有效")
    created_at = Column(DateTime, default=datetime.now, index=True, comment="创建时间")
    executed_at = Column(DateTime, comment="执行时间")
    closed_at = Column(DateTime, comment="平仓时间")

    __table_args__ = (
        Index("idx_strategy_active", "strategy_id", "is_active"),
        Index("idx_symbol_signal", "symbol", "is_active"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class StockPool(Base):
    """Stock pool for tracking tradable symbols."""
    __tablename__ = "stock_pool"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False, comment="股票代码")
    name = Column(String(50), comment="股票名称")
    sector = Column(String(50), comment="所属行业")
    index_name = Column(String(50), comment="指数名称(如沪深300)")
    is_active = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    __table_args__ = (
        Index("idx_pool_active", "is_active"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class MarketStatus(Base):
    """Market status and data update tracking."""
    __tablename__ = "market_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    data_type = Column(String(50), nullable=False, comment="数据类型")
    last_update = Column(DateTime, comment="最后更新时间")
    status = Column(String(20), default="pending", comment="状态: pending/running/success/failed")
    error_message = Column(Text, comment="错误信息")
    created_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="创建时间")

    __table_args__ = (
        Index("idx_data_type", "data_type"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class RealtimePrice(Base):
    """Real-time stock prices from spot data."""
    __tablename__ = "realtime_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), unique=True, nullable=False, comment="股票代码")
    price = Column(Float, nullable=False, comment="最新价")
    change = Column(Float, comment="涨跌额")
    change_pct = Column(Float, comment="涨跌幅(%)")
    volume = Column(Float, comment="成交量(手)")
    amount = Column(Float, comment="成交额(元)")
    high = Column(Float, comment="最高价")
    low = Column(Float, comment="最低价")
    open = Column(Float, comment="今开价")
    prev_close = Column(Float, comment="昨收价")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    __table_args__ = (
        Index("idx_realtime_symbol", "symbol"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class StrategyPosition(Base):
    """Current strategy positions for daily holdings display."""
    __tablename__ = "strategy_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), nullable=False, comment="策略ID")
    symbol = Column(String(10), nullable=False, comment="股票代码")
    sector = Column(String(50), comment="行业板块")
    direction = Column(String(10), default="Long", comment="方向(Long/Short)")
    quantity = Column(Integer, default=0, comment="持仓数量")
    entry_price = Column(Float, comment="入场价格")
    current_price = Column(Float, comment="当前价格")
    days_held = Column(Integer, default=0, comment="持仓天数")
    weight = Column(Float, comment="仓位占比(%)")
    floating_pnl = Column(Float, default=0, comment="浮盈浮亏(%)")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    __table_args__ = (
        Index("idx_strategy_position", "strategy_id", "symbol"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class TushareStockDaily(Base):
    """Tushare日线数据（分区表）"""
    __tablename__ = "tushare_stock_daily"

    ts_code = Column(String(10), primary_key=True, comment="tushare股票代码")
    trade_date = Column(Date, primary_key=True, comment="交易日期")
    open = Column(Float, comment="开盘价")
    high = Column(Float, comment="最高价")
    low = Column(Float, comment="最低价")
    close = Column(Float, comment="收盘价")
    pre_close = Column(Float, comment="昨收价")
    change_amt = Column(Float, comment="涨跌额")
    pct_chg = Column(Float, comment="涨跌幅(%)")
    vol = Column(Float, comment="成交量(手)")
    amount = Column(Float, comment="成交额(千元)")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    __table_args__ = (
        Index("idx_tushare_trade_date", "trade_date"),
        Index("idx_tushare_ts_code", "ts_code"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class TushareStockLimit(Base):
    """Tushare涨跌停价格数据"""
    __tablename__ = "tushare_stock_limit"

    ts_code = Column(String(10), primary_key=True, comment="tushare股票代码")
    trade_date = Column(Date, primary_key=True, comment="交易日期")
    pre_close = Column(Float, comment="昨日收盘价")
    up_limit = Column(Float, comment="涨停价")
    down_limit = Column(Float, comment="跌停价")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    __table_args__ = (
        Index("idx_limit_trade_date", "trade_date"),
        Index("idx_limit_ts_code", "ts_code"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


class TushareStockBasic(Base):
    """Tushare股票基础信息"""
    __tablename__ = "tushare_stock_basic"

    ts_code = Column(String(10), primary_key=True, comment="tushare股票代码")
    symbol = Column(String(6), comment="股票代码")
    name = Column(String(50), comment="股票名称")
    area = Column(String(20), comment="地域")
    industry = Column(String(50), comment="所属行业")
    market = Column(String(10), comment="市场类型(主板/创业板/科创板)")
    list_date = Column(Date, comment="上市日期")
    list_status = Column(String(1), comment="上市状态(L上市/D退市/P暂停)")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    __table_args__ = (
        Index("idx_basic_symbol", "symbol"),
        Index("idx_basic_industry", "industry"),
        {"mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
    )


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
