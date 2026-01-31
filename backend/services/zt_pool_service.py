"""
涨停股池数据服务 (ZT Pool Service)

提供涨停/跌停数据获取、缓存、交易日历管理功能。
用于龙厂策略 (Dragon Leader Strategy) 的数据支持。
"""
import os
import sys
import time
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Set, Tuple
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Patch AkShare before importing
from utils.patch_akshare import ensure_patched
ensure_patched()

import akshare as ak

from database.connection import get_session
from database.models import StockDaily
from services.data_service import StockDataService
from utils.rate_limiter import get_akshare_limiter

logger = logging.getLogger(__name__)


class ZTPoolService:
    """
    涨停股池数据服务

    Features:
    - 交易日历管理（从数据库提取）
    - 涨停/跌停数据获取和缓存
    - 提取所有候选股票代码
    - 增量加载股票行情数据
    """

    def __init__(self):
        self.rate_limiter = get_akshare_limiter()
        self.data_service = StockDataService()

        # 缓存
        self._trading_days_cache: List[date] = []
        self._zt_pool_cache: Dict[date, pd.DataFrame] = {}  # 涨停股池缓存
        self._dt_pool_cache: Dict[date, pd.DataFrame] = {}  # 跌停股池缓存

    def get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """
        获取交易日历

        从数据库 stock_daily 表提取已有交易日。

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表（升序）
        """
        if self._trading_days_cache:
            # 从缓存筛选
            return [d for d in self._trading_days_cache if start_date <= d <= end_date]

        with get_session() as session:
            from sqlalchemy import func, distinct
            results = session.query(
                distinct(StockDaily.trade_date)
            ).filter(
                StockDaily.trade_date >= start_date,
                StockDaily.trade_date <= end_date
            ).order_by(StockDaily.trade_date).all()

            trading_days = [r[0] for r in results]
            self._trading_days_cache = trading_days
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
        # 确保交易日历已加载
        if not self._trading_days_cache:
            # 加载更大范围的交易日历
            self.get_trading_days(current_date - timedelta(days=365), current_date)

        # 找到当前日期或之前最近的交易日
        prev_days = [d for d in self._trading_days_cache if d < current_date]

        if len(prev_days) >= n:
            return prev_days[-n]
        return None

    def fetch_zt_pool(self, trade_date: date) -> Optional[pd.DataFrame]:
        """
        获取指定日期的涨停股池

        Args:
            trade_date: 交易日期

        Returns:
            涨停股池 DataFrame，包含字段：代码, 名称, 连板数, 炸板次数, 首次封板时间 等
        """
        # 检查缓存
        if trade_date in self._zt_pool_cache:
            return self._zt_pool_cache[trade_date]

        date_str = trade_date.strftime("%Y%m%d")

        try:
            self.rate_limiter.acquire(blocking=True)
            df = ak.stock_zt_pool_em(date=date_str)

            if df is not None and not df.empty:
                self._zt_pool_cache[trade_date] = df
                logger.debug(f"Fetched ZT pool for {date_str}: {len(df)} stocks")
                return df
            else:
                # 空数据也缓存，避免重复请求
                self._zt_pool_cache[trade_date] = pd.DataFrame()
                return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Failed to fetch ZT pool for {date_str}: {e}")
            self._zt_pool_cache[trade_date] = pd.DataFrame()
            return pd.DataFrame()

    def fetch_dt_pool(self, trade_date: date) -> Optional[pd.DataFrame]:
        """
        获取指定日期的跌停股池

        Args:
            trade_date: 交易日期

        Returns:
            跌停股池 DataFrame
        """
        # 检查缓存
        if trade_date in self._dt_pool_cache:
            return self._dt_pool_cache[trade_date]

        date_str = trade_date.strftime("%Y%m%d")

        try:
            self.rate_limiter.acquire(blocking=True)
            df = ak.stock_zt_pool_dtgc_em(date=date_str)

            if df is not None and not df.empty:
                self._dt_pool_cache[trade_date] = df
                logger.debug(f"Fetched DT pool for {date_str}: {len(df)} stocks")
                return df
            else:
                self._dt_pool_cache[trade_date] = pd.DataFrame()
                return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Failed to fetch DT pool for {date_str}: {e}")
            self._dt_pool_cache[trade_date] = pd.DataFrame()
            return pd.DataFrame()

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
            # 涨停池
            zt_df = self.fetch_zt_pool(trade_date)
            if zt_df is not None and not zt_df.empty:
                zt_count += 1

            # 跌停池
            dt_df = self.fetch_dt_pool(trade_date)
            if dt_df is not None and not dt_df.empty:
                dt_count += 1

            # 进度日志
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(trading_days)} days loaded")

            # 请求间隔
            time.sleep(0.3)

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

    def load_symbols_incremental(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, int]:
        """
        增量加载股票行情数据

        检查本地数据库，只下载缺失的数据。

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            {'loaded': 已加载数, 'skipped': 跳过数, 'failed': 失败数}
        """
        stats = {'loaded': 0, 'skipped': 0, 'failed': 0}

        logger.info(f"Incremental loading {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols):
            try:
                # 检查本地数据覆盖范围
                with get_session() as session:
                    from sqlalchemy import func
                    result = session.query(
                        func.min(StockDaily.trade_date),
                        func.max(StockDaily.trade_date)
                    ).filter(StockDaily.symbol == symbol).first()

                    local_min, local_max = result if result else (None, None)

                # 判断是否需要加载
                if local_min and local_max:
                    if local_min <= start_date and local_max >= end_date:
                        # 本地数据完整覆盖，跳过
                        stats['skipped'] += 1
                        continue

                # 需要下载数据
                start_str = start_date.strftime("%Y%m%d")
                end_str = end_date.strftime("%Y%m%d")

                df = self.data_service.fetch_stock_data(symbol, start_str, end_str)

                if df is not None and not df.empty:
                    from database.repository import StockDataRepository
                    count = StockDataRepository.save_stock_data(symbol, df)
                    stats['loaded'] += 1
                    logger.debug(f"[{i+1}/{len(symbols)}] Loaded {symbol}: {count} records")
                else:
                    stats['failed'] += 1
                    logger.warning(f"[{i+1}/{len(symbols)}] Failed to load {symbol}")

                # 请求间隔
                time.sleep(0.5)

            except Exception as e:
                stats['failed'] += 1
                logger.error(f"Error loading {symbol}: {e}")

            # 进度日志
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i+1}/{len(symbols)} symbols processed")

        logger.info(f"Incremental load complete: {stats}")
        return stats

    def prepare_for_backtest(self, start_date: date, end_date: date) -> Dict:
        """
        为回测准备所有必要数据

        完整的预加载流程：
        1. 获取交易日历
        2. 预加载涨停/跌停数据
        3. 提取所有候选股票代码
        4. 增量加载股票行情数据

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            准备结果统计
        """
        logger.info(f"=== Preparing data for backtest: {start_date} to {end_date} ===")

        result = {
            'trading_days': 0,
            'zt_days': 0,
            'dt_days': 0,
            'symbols_count': 0,
            'load_stats': {}
        }

        # Step 1: 获取交易日历
        trading_days = self.get_trading_days(start_date, end_date)
        result['trading_days'] = len(trading_days)
        logger.info(f"Step 1: Found {len(trading_days)} trading days")

        # Step 2: 预加载涨停/跌停数据
        zt_count, dt_count = self.preload_zt_dt_data(start_date, end_date)
        result['zt_days'] = zt_count
        result['dt_days'] = dt_count
        logger.info(f"Step 2: Loaded ZT/DT data for {zt_count}/{dt_count} days")

        # Step 3: 提取所有候选股票代码
        all_symbols = self.extract_all_symbols()
        result['symbols_count'] = len(all_symbols)
        logger.info(f"Step 3: Extracted {len(all_symbols)} unique symbols")

        # Step 4: 增量加载股票行情数据
        if all_symbols:
            load_stats = self.load_symbols_incremental(
                list(all_symbols), start_date, end_date
            )
            result['load_stats'] = load_stats
            logger.info(f"Step 4: Load stats - {load_stats}")

        logger.info(f"=== Backtest data preparation complete ===")
        return result

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

        # 筛选连板数 >= min_board
        if '连板数' not in df.columns:
            return []

        filtered = df[df['连板数'] >= min_board].copy()

        if filtered.empty:
            return []

        # 按连板数降序排序
        filtered = filtered.sort_values('连板数', ascending=False)

        # 转换为字典列表
        result = []
        for _, row in filtered.iterrows():
            result.append({
                'symbol': row['代码'],
                'name': row['名称'],
                'board_count': int(row['连板数']),
                'break_count': int(row.get('炸板次数', 0)),
                'first_board_time': row.get('首次封板时间', ''),
                'industry': row.get('所属行业', '')
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
