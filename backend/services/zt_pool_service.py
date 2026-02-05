"""
涨停股池数据服务 (ZT Pool Service)

基于 Tushare 数据库计算涨停/跌停股票，支持：
- 涨停股识别（收盘价 >= 涨停价 * 0.998）
- 跌停股识别（收盘价 <= 跌停价 * 1.002）
- 连板数计算
- 一字板判断（开盘价 = 最高价 = 最低价 = 收盘价 ≈ 涨停价）
- 交易日历管理
"""
import os
import sys
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Set, Tuple
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database.connection import get_session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# 涨停判断阈值（允许0.2%误差）
LIMIT_UP_THRESHOLD = 0.998
LIMIT_DOWN_THRESHOLD = 1.002
# 一字板判断阈值（OHLC差异小于0.1%）
YIZI_THRESHOLD = 0.001


class ZTPoolService:
    """
    涨停股池数据服务

    Features:
    - 交易日历管理（从数据库提取）
    - 涨停/跌停股票计算
    - 连板数计算
    - 一字板判断
    """

    def __init__(self):
        # 缓存
        self._trading_days_cache: List[date] = []
        self._zt_pool_cache: Dict[date, pd.DataFrame] = {}
        self._dt_pool_cache: Dict[date, pd.DataFrame] = {}

    def get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """
        获取交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表（升序）
        """
        if self._trading_days_cache:
            filtered = [d for d in self._trading_days_cache if start_date <= d <= end_date]
            if filtered:
                return filtered

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
            trading_days = [row[0] for row in result.fetchall()]

            # 更新缓存
            if not self._trading_days_cache:
                self._trading_days_cache = trading_days
            else:
                all_days = set(self._trading_days_cache) | set(trading_days)
                self._trading_days_cache = sorted(all_days)

            logger.info(f"Loaded {len(trading_days)} trading days from database")
            return trading_days

    def get_previous_trading_day(self, current_date: date, n: int = 1) -> Optional[date]:
        """
        获取前n个交易日

        Args:
            current_date: 当前日期
            n: 往前推几个交易日

        Returns:
            交易日日期，如果不存在返回 None
        """
        if not self._trading_days_cache:
            self.get_trading_days(current_date - timedelta(days=365), current_date)

        prev_days = [d for d in self._trading_days_cache if d < current_date]

        if len(prev_days) >= n:
            return prev_days[-n]
        return None

    def fetch_zt_pool(self, trade_date: date) -> pd.DataFrame:
        """
        获取指定日期的涨停股池

        通过比较收盘价和涨停价来判断是否涨停

        Args:
            trade_date: 交易日期

        Returns:
            涨停股池 DataFrame，包含字段：
            - ts_code: 股票代码
            - name: 股票名称
            - close: 收盘价
            - up_limit: 涨停价
            - pct_chg: 涨跌幅
            - board_count: 连板数
            - is_yizi: 是否一字板
            - industry: 所属行业
        """
        if trade_date in self._zt_pool_cache:
            return self._zt_pool_cache[trade_date]

        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT
                        d.ts_code,
                        b.name,
                        d.open,
                        d.high,
                        d.low,
                        d.close,
                        l.up_limit,
                        l.down_limit,
                        d.pct_chg,
                        b.industry
                    FROM tushare_stock_daily d
                    JOIN tushare_stock_limit l
                        ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
                    LEFT JOIN tushare_stock_basic b
                        ON d.ts_code = b.ts_code
                    WHERE d.trade_date = :trade_date
                    AND d.close >= l.up_limit * :threshold
                """),
                {'trade_date': trade_date, 'threshold': LIMIT_UP_THRESHOLD}
            )
            rows = result.fetchall()

        if not rows:
            self._zt_pool_cache[trade_date] = pd.DataFrame()
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            'ts_code', 'name', 'open', 'high', 'low', 'close',
            'up_limit', 'down_limit', 'pct_chg', 'industry'
        ])

        # 判断一字板：开盘价 = 最高价 = 最低价 = 收盘价 ≈ 涨停价
        df['is_yizi'] = df.apply(self._is_yizi_board, axis=1)

        # 计算连板数
        df['board_count'] = df['ts_code'].apply(
            lambda x: self._calculate_board_count(x, trade_date)
        )

        # 转换列名以兼容旧接口
        df['代码'] = df['ts_code'].str.replace(r'\.(SZ|SH)$', '', regex=True)
        df['名称'] = df['name']
        df['连板数'] = df['board_count']
        df['一字板'] = df['is_yizi']
        df['所属行业'] = df['industry']

        self._zt_pool_cache[trade_date] = df
        logger.debug(f"Calculated ZT pool for {trade_date}: {len(df)} stocks")

        return df

    def fetch_dt_pool(self, trade_date: date) -> pd.DataFrame:
        """
        获取指定日期的跌停股池

        Args:
            trade_date: 交易日期

        Returns:
            跌停股池 DataFrame
        """
        if trade_date in self._dt_pool_cache:
            return self._dt_pool_cache[trade_date]

        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT
                        d.ts_code,
                        b.name,
                        d.close,
                        l.down_limit,
                        d.pct_chg,
                        b.industry
                    FROM tushare_stock_daily d
                    JOIN tushare_stock_limit l
                        ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
                    LEFT JOIN tushare_stock_basic b
                        ON d.ts_code = b.ts_code
                    WHERE d.trade_date = :trade_date
                    AND d.close <= l.down_limit * :threshold
                """),
                {'trade_date': trade_date, 'threshold': LIMIT_DOWN_THRESHOLD}
            )
            rows = result.fetchall()

        if not rows:
            self._dt_pool_cache[trade_date] = pd.DataFrame()
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            'ts_code', 'name', 'close', 'down_limit', 'pct_chg', 'industry'
        ])

        df['代码'] = df['ts_code'].str.replace(r'\.(SZ|SH)$', '', regex=True)
        df['名称'] = df['name']

        self._dt_pool_cache[trade_date] = df
        logger.debug(f"Calculated DT pool for {trade_date}: {len(df)} stocks")

        return df

    def _is_yizi_board(self, row) -> bool:
        """
        判断是否为一字板

        一字板条件：开盘价 = 最高价 = 最低价 = 收盘价 ≈ 涨停价
        """
        if row['up_limit'] is None or row['up_limit'] == 0:
            return False

        prices = [row['open'], row['high'], row['low'], row['close']]
        if any(p is None for p in prices):
            return False

        # 检查OHLC是否相等（允许微小误差）
        max_price = max(prices)
        min_price = min(prices)

        if max_price == 0:
            return False

        price_diff = (max_price - min_price) / max_price

        # OHLC差异小于阈值，且收盘价接近涨停价
        return price_diff < YIZI_THRESHOLD and row['close'] >= row['up_limit'] * LIMIT_UP_THRESHOLD

    def _calculate_board_count(self, ts_code: str, trade_date: date) -> int:
        """
        计算连板数

        从当天往前回溯，统计连续涨停天数

        Args:
            ts_code: 股票代码
            trade_date: 交易日期

        Returns:
            连板数
        """
        with get_session() as session:
            # 获取最近30个交易日的数据（足够计算连板）
            result = session.execute(
                text("""
                    SELECT d.trade_date, d.close, l.up_limit
                    FROM tushare_stock_daily d
                    JOIN tushare_stock_limit l
                        ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
                    WHERE d.ts_code = :ts_code
                    AND d.trade_date <= :trade_date
                    ORDER BY d.trade_date DESC
                    LIMIT 30
                """),
                {'ts_code': ts_code, 'trade_date': trade_date}
            )
            rows = result.fetchall()

        if not rows:
            return 0

        board_count = 0
        for row in rows:
            close_price = row[1]
            up_limit = row[2]

            if up_limit is None or up_limit == 0:
                break

            if close_price >= up_limit * LIMIT_UP_THRESHOLD:
                board_count += 1
            else:
                break

        return board_count

    def preload_zt_dt_data(self, start_date: date, end_date: date) -> Tuple[int, int]:
        """
        预加载回测期间的涨停/跌停数据

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            (涨停池加载天数, 跌停池加载天数)
        """
        trading_days = self.get_trading_days(start_date, end_date)

        zt_count = 0
        dt_count = 0

        logger.info(f"Preloading ZT/DT data for {len(trading_days)} trading days...")

        for i, trade_date in enumerate(trading_days):
            zt_df = self.fetch_zt_pool(trade_date)
            if zt_df is not None and not zt_df.empty:
                zt_count += 1

            dt_df = self.fetch_dt_pool(trade_date)
            if dt_df is not None and not dt_df.empty:
                dt_count += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(trading_days)} days loaded")

        logger.info(f"Preloaded ZT data: {zt_count} days, DT data: {dt_count} days")
        return zt_count, dt_count

    def extract_all_symbols(self) -> Set[str]:
        """
        从已缓存的涨停池中提取所有出现过的股票代码

        Returns:
            股票代码集合
        """
        all_symbols = set()

        for trade_date, df in self._zt_pool_cache.items():
            if df is not None and not df.empty and '代码' in df.columns:
                symbols = df['代码'].tolist()
                all_symbols.update(symbols)

        logger.info(f"Extracted {len(all_symbols)} unique symbols from ZT pool cache")
        return all_symbols

    def get_max_board_stocks(self, trade_date: date, min_board: int = 3) -> List[Dict]:
        """
        获取指定日期的最高板股票

        Args:
            trade_date: 交易日期（应该是T-1日）
            min_board: 最低连板数要求

        Returns:
            最高板股票列表，按连板数降序
        """
        df = self.fetch_zt_pool(trade_date)

        if df is None or df.empty:
            return []

        if '连板数' not in df.columns:
            return []

        filtered = df[df['连板数'] >= min_board].copy()

        if filtered.empty:
            return []

        filtered = filtered.sort_values('连板数', ascending=False)

        result = []
        for _, row in filtered.iterrows():
            result.append({
                'symbol': row['代码'],
                'ts_code': row['ts_code'],
                'name': row['名称'],
                'board_count': int(row['连板数']),
                'is_yizi': bool(row.get('一字板', False)),
                'industry': row.get('所属行业', ''),
                'pct_chg': row.get('pct_chg', 0)
            })

        return result

    def get_dt_count(self, trade_date: date) -> int:
        """
        获取指定日期的跌停股数量

        Args:
            trade_date: 交易日期

        Returns:
            跌停股数量
        """
        df = self.fetch_dt_pool(trade_date)

        if df is None or df.empty:
            return 0

        return len(df)

    def clear_cache(self):
        """清空所有缓存"""
        self._trading_days_cache = []
        self._zt_pool_cache = {}
        self._dt_pool_cache = {}
        logger.info("ZT Pool Service cache cleared")


# 单例
_zt_pool_service: Optional[ZTPoolService] = None


def get_zt_pool_service() -> ZTPoolService:
    """获取涨停股池服务单例"""
    global _zt_pool_service
    if _zt_pool_service is None:
        _zt_pool_service = ZTPoolService()
    return _zt_pool_service
