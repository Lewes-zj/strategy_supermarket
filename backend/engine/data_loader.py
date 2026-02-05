"""
数据加载器 - 从 Tushare 数据库表读取股票数据

提供统一的数据访问接口，支持：
- 单只股票日线数据查询
- 多只股票批量查询
- 股票池管理
"""
import pandas as pd
from datetime import datetime, date
from typing import List, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database.connection import get_session
from sqlalchemy import text


def fetch_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq"
) -> pd.DataFrame:
    """
    从数据库获取单只股票的日线数据

    Args:
        symbol: 股票代码，支持两种格式：
                - tushare格式: "000001.SZ"
                - 纯代码格式: "000001"
        start_date: 开始日期 "YYYYMMDD"
        end_date: 结束日期 "YYYYMMDD"
        adjust: 复权类型（当前数据库存储的是不复权数据）

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, symbol
    """
    ts_code = _normalize_ts_code(symbol)

    start_dt = datetime.strptime(start_date, "%Y%m%d").date()
    end_dt = datetime.strptime(end_date, "%Y%m%d").date()

    with get_session() as session:
        result = session.execute(
            text("""
                SELECT trade_date, open, high, low, close, vol as volume,
                       amount, pre_close, change_amt, pct_chg, ts_code
                FROM tushare_stock_daily
                WHERE ts_code = :ts_code
                AND trade_date BETWEEN :start_date AND :end_date
                ORDER BY trade_date
            """),
            {'ts_code': ts_code, 'start_date': start_dt, 'end_date': end_dt}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        'date', 'open', 'high', 'low', 'close', 'volume',
        'amount', 'pre_close', 'change_amt', 'pct_chg', 'symbol'
    ])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df


def fetch_multiple_symbols(
    symbols: List[str],
    start_date: str,
    end_date: str = None,
    adjust: str = "qfq"
) -> pd.DataFrame:
    """
    批量获取多只股票的日线数据

    Args:
        symbols: 股票代码列表
        start_date: 开始日期 "YYYYMMDD"
        end_date: 结束日期 "YYYYMMDD"
        adjust: 复权类型

    Returns:
        DataFrame with symbol column
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    ts_codes = [_normalize_ts_code(s) for s in symbols]
    start_dt = datetime.strptime(start_date, "%Y%m%d").date()
    end_dt = datetime.strptime(end_date, "%Y%m%d").date()

    with get_session() as session:
        placeholders = ','.join([f':code_{i}' for i in range(len(ts_codes))])
        params = {f'code_{i}': code for i, code in enumerate(ts_codes)}
        params['start_date'] = start_dt
        params['end_date'] = end_dt

        result = session.execute(
            text(f"""
                SELECT trade_date, open, high, low, close, vol as volume,
                       amount, pre_close, change_amt, pct_chg, ts_code
                FROM tushare_stock_daily
                WHERE ts_code IN ({placeholders})
                AND trade_date BETWEEN :start_date AND :end_date
                ORDER BY ts_code, trade_date
            """),
            params
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        'date', 'open', 'high', 'low', 'close', 'volume',
        'amount', 'pre_close', 'change_amt', 'pct_chg', 'symbol'
    ])
    df['date'] = pd.to_datetime(df['date'])

    return df


def get_stock_pool() -> List[str]:
    """获取活跃股票池（从数据库中有数据的股票）"""
    with get_session() as session:
        result = session.execute(
            text("SELECT DISTINCT ts_code FROM tushare_stock_daily ORDER BY ts_code")
        )
        return [row[0] for row in result.fetchall()]


def get_stock_pool_details() -> List[dict]:
    """获取股票池详细信息"""
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT ts_code, symbol, name, industry, market, list_date
                FROM tushare_stock_basic
                WHERE list_status = 'L'
                ORDER BY ts_code
            """)
        )
        rows = result.fetchall()

    return [
        {
            'ts_code': row[0],
            'symbol': row[1],
            'name': row[2],
            'industry': row[3],
            'market': row[4],
            'list_date': row[5]
        }
        for row in rows
    ]


def get_trading_days(start_date: date, end_date: date) -> List[date]:
    """
    从数据库获取交易日历

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        交易日列表（升序）
    """
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT DISTINCT trade_date
                FROM tushare_stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                ORDER BY trade_date
            """),
            {'start_date': start_date, 'end_date': end_date}
        )
        return [row[0] for row in result.fetchall()]


def _normalize_ts_code(symbol: str) -> str:
    """
    标准化股票代码为 tushare 格式

    Args:
        symbol: 输入代码，如 "000001" 或 "000001.SZ"

    Returns:
        tushare格式代码，如 "000001.SZ"
    """
    if '.' in symbol:
        return symbol.upper()

    # 根据代码前缀判断交易所
    if symbol.startswith(('6', '9')):
        return f"{symbol}.SH"
    else:
        return f"{symbol}.SZ"
