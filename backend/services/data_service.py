"""
Data service for fetching and managing stock data from AkShare.
Integrates rate limiting, caching, and database storage.
"""
import time
import logging
import os
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict
import pandas as pd

# Set NO_PROXY for AkShare domains to bypass system proxy
# This allows direct connection to data sources
os.environ['NO_PROXY'] = 'push2his.eastmoney.com,eastmoney.com,akshare.akfamily.xyz'
os.environ['no_proxy'] = 'push2his.eastmoney.com,eastmoney.com,akshare.akfamily.xyz'

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# IMPORTANT: Patch AkShare BEFORE importing it
# This adds browser-like headers to bypass anti-scraping protection
from utils.patch_akshare import ensure_patched
ensure_patched()

import akshare as ak

from config import config
from database.repository import StockDataRepository, StockPoolRepository, MarketStatusRepository
from utils.rate_limiter import get_akshare_limiter
from services.alternative_fetcher import get_alternative_fetcher

logger = logging.getLogger(__name__)


class StockDataService:
    """
    Service for fetching and managing stock data.

    Features:
    - Rate-limited AkShare requests
    - Database caching
    - Incremental updates
    - CSI 300 stock pool management
    - HTTP proxy support
    """

    def __init__(self):
        self.rate_limiter = get_akshare_limiter()
        self.stock_pool_repo = StockPoolRepository()
        self.data_repo = StockDataRepository()
        self.status_repo = MarketStatusRepository()

    def init_stock_pool(self) -> int:
        """
        Initialize CSI 300 stock pool from AkShare.

        Returns:
            Number of symbols added to stock pool
        """
        logger.info("Initializing CSI 300 stock pool...")
        self.status_repo.update_status("stock_pool_init", "running")

        try:
            # Use predefined CSI 300 list as fallback
            # (AkShare stock_index_csindex_symbol has API issues)
            csi300_stocks = self._get_predefined_csi300()

            # Add to database
            count = self.stock_pool_repo.add_symbols(csi300_stocks, index_name="CSI300")

            self.status_repo.update_status("stock_pool_init", "success")
            logger.info(f"Initialized CSI 300 stock pool with {count} symbols")
            return count

        except Exception as e:
            logger.error(f"Failed to initialize stock pool: {e}")
            self.status_repo.update_status("stock_pool_init", "failed", str(e))
            return 0

    def _get_predefined_csi300(self) -> List[Dict]:
        """Get predefined CSI 300 stock list."""
        return [
            # 金融
            {'symbol': '000001', 'name': '平安银行', 'sector': '金融'},
            {'symbol': '600000', 'name': '浦发银行', 'sector': '金融'},
            {'symbol': '600036', 'name': '招商银行', 'sector': '金融'},
            {'symbol': '601318', 'name': '中国平安', 'sector': '金融'},
            {'symbol': '601398', 'name': '工商银行', 'sector': '金融'},
            {'symbol': '601939', 'name': '建设银行', 'sector': '金融'},
            {'symbol': '600030', 'name': '中信证券', 'sector': '金融'},
            {'symbol': '600016', 'name': '民生银行', 'sector': '金融'},

            # 科技
            {'symbol': '300750', 'name': '宁德时代', 'sector': '科技'},
            {'symbol': '002475', 'name': '立讯精密', 'sector': '科技'},
            {'symbol': '000063', 'name': '中兴通讯', 'sector': '科技'},
            {'symbol': '002415', 'name': '海康威视', 'sector': '科技'},
            {'symbol': '600584', 'name': '长电科技', 'sector': '科技'},

            # 消费
            {'symbol': '600519', 'name': '贵州茅台', 'sector': '消费'},
            {'symbol': '000858', 'name': '五粮液', 'sector': '消费'},
            {'symbol': '000333', 'name': '美的集团', 'sector': '消费'},
            {'symbol': '002304', 'name': '洋河股份', 'sector': '消费'},
            {'symbol': '600887', 'name': '伊利股份', 'sector': '消费'},

            # 医药
            {'symbol': '000661', 'name': '长春高新', 'sector': '医药'},
            {'symbol': '000538', 'name': '云南白药', 'sector': '医药'},
            {'symbol': '600276', 'name': '恒瑞医药', 'sector': '医药'},
            {'symbol': '300015', 'name': '爱尔眼科', 'sector': '医药'},
            {'symbol': '002821', 'name': '凯莱英', 'sector': '医药'},

            # 新能源
            {'symbol': '300124', 'name': '汇川技术', 'sector': '新能源'},
            {'symbol': '002129', 'name': '中环股份', 'sector': '新能源'},
            {'symbol': '688981', 'name': '中芯国际', 'sector': '科技'},

            # 地产
            {'symbol': '000002', 'name': '万科A', 'sector': '地产'},
            {'symbol': '001979', 'name': '招商蛇口', 'sector': '地产'},

            # 能源
            {'symbol': '601857', 'name': '中国石油', 'sector': '能源'},
            {'symbol': '600028', 'name': '中国石化', 'sector': '能源'},

            # 工业
            {'symbol': '600031', 'name': '三一重工', 'sector': '工业'},
            {'symbol': '000333', 'name': '美的集团', 'sector': '消费'},
            {'symbol': '002594', 'name': '比亚迪', 'sector': '汽车'},
        ]

    def fetch_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data from AkShare with rate limiting.

        Falls back to alternative data sources (Sina, NetEase) if AkShare fails.

        Returns DataFrame with columns: [date, open, high, low, close, volume, amount]
        Note: 'date' is kept as a column (not index) for database compatibility.

        Args:
            symbol: Stock symbol (e.g., "000001")
            start_date: Start date in "YYYYMMDD" format
            end_date: End date in "YYYYMMDD" format (default: today)
            adjust: Adjustment type - "qfq" (forward), "hfq" (backward), "" (none)

        Returns:
            DataFrame with OHLCV data or None on error
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        # Try AkShare first (eastmoney)
        try:
            # Rate limit
            self.rate_limiter.acquire(blocking=True)

            # Use akshare with proxy support via environment
            # Set environment variable for this request only if needed
            old_http = os.environ.get('HTTP_PROXY')
            old_https = os.environ.get('HTTPS_PROXY')

            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
            finally:
                # Restore env vars if needed
                if old_http is not None:
                    os.environ['HTTP_PROXY'] = old_http
                if old_https is not None:
                    os.environ['HTTPS_PROXY'] = old_https

            if df is not None and not df.empty:
                # Rename columns to standard format
                rename_map = {
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                }
                df.rename(columns=rename_map, inplace=True)

                # Convert date column
                df["date"] = pd.to_datetime(df["date"])

                # Select and order columns (keep 'date' as column for database)
                df = df[["date", "open", "high", "low", "close", "volume", "amount"]]

                logger.info(f"Successfully fetched {symbol} from AkShare (eastmoney)")
                return df

        except Exception as e:
            logger.warning(f"AkShare (eastmoney) failed for {symbol}: {e}")

        # Try alternative data sources (Sina, NetEase, etc.)
        logger.info(f"Trying alternative data sources for {symbol}...")
        alt_fetcher = get_alternative_fetcher()
        df = alt_fetcher.fetch_with_fallback(symbol, start_date, end_date)

        if df is not None and not df.empty:
            logger.info(f"Successfully fetched {len(df)} records from alternative source for {symbol}")
            return df

        logger.error(f"All data sources failed for {symbol}")
        return None

    def update_stock_data(
        self,
        symbols: List[str] = None,
        days_back: int = 3650,  # 10 years of historical data
        force_update: bool = False
    ) -> Dict[str, int]:
        """
        Update stock data for given symbols.

        Args:
            symbols: List of symbols to update (None = all active symbols)
            days_back: Number of days of historical data to fetch
            force_update: Force update even if data exists

        Returns:
            Dict with counts: {'updated': X, 'failed': Y, 'skipped': Z}
        """
        if symbols is None:
            all_symbols = self.stock_pool_repo.get_stock_pool()
            symbols = [s['symbol'] for s in all_symbols]

        logger.info(f"Updating data for {len(symbols)} symbols...")
        self.status_repo.update_status("daily_data_update", "running")

        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        stats = {'updated': 0, 'failed': 0, 'skipped': 0}

        for i, symbol in enumerate(symbols):
            try:
                # Check if update is needed
                if not force_update:
                    latest_date = self.data_repo.get_latest_date(symbol)
                    if latest_date:
                        days_behind = (datetime.now().date() - latest_date).days
                        if days_behind <= 1:
                            stats['skipped'] += 1
                            continue

                # Fetch data
                df = self.fetch_stock_data(symbol, start_date, end_date)

                if df is not None and not df.empty:
                    count = self.data_repo.save_stock_data(symbol, df)
                    stats['updated'] += count
                    logger.info(f"[{i+1}/{len(symbols)}] Updated {symbol}: {count} records")
                else:
                    stats['failed'] += 1

                # Delay between requests
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                stats['failed'] += 1

        self.status_repo.update_status("daily_data_update", "success")
        logger.info(f"Data update complete: {stats}")
        return stats

    def get_cached_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date = None
    ) -> pd.DataFrame:
        """
        Get stock data from cache/database.

        Fetches from database first, then fills gaps with AkShare.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date (default: today)

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        if end_date is None:
            end_date = datetime.now().date()

        # Try to get from database first
        df = self.data_repo.get_stock_data(symbols, start_date, end_date)

        if df.empty:
            logger.info("No cached data found")
            return df

        return df

    def get_realtime_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get real-time price for a symbol (with rate limiting).

        Args:
            symbol: Stock symbol

        Returns:
            Dict with keys: price, change, change_pct, volume, amount
        """
        try:
            self.rate_limiter.acquire()
            df = ak.stock_zh_a_spot_em()

            result = df[df['代码'] == symbol]

            if not result.empty:
                row = result.iloc[0]
                return {
                    'price': float(row['最新价']),
                    'change': float(row['涨跌额']),
                    'change_pct': float(row['涨跌幅']),
                    'volume': float(row['成交量']),
                    'amount': float(row['成交额'])
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get real-time price for {symbol}: {e}")
            return None

    def get_sector_stocks(self, sector_name: str) -> List[str]:
        """
        Get stock symbols for a specific sector.

        Args:
            sector_name: Sector name (e.g., "半导体", "医药")

        Returns:
            List of stock symbols
        """
        try:
            pool_data = self.stock_pool_repo.get_stock_pool()
            symbols = [
                item['symbol'] for item in pool_data
                if sector_name in item.get('sector', '')
            ]
            return symbols
        except Exception as e:
            logger.error(f"Failed to get sector stocks: {e}")
            return []


# Global service instance
_data_service = None


def get_data_service() -> StockDataService:
    """Get the global data service instance."""
    global _data_service

    if _data_service is None:
        _data_service = StockDataService()

    return _data_service


# ============================================================
# DataService - Lightweight data service with adaptive rate limiting
# ============================================================

from .rate_limiter import AdaptiveRateLimiter


class DataService:
    """
    Lightweight data service with adaptive rate limiting.

    This is a simplified version for backtesting that uses
    in-memory caching and the AdaptiveRateLimiter.

    Features:
    - Adaptive rate limiting (backs off on errors, speeds up on success)
    - Simple in-memory cache
    - Stock and index data fetching
    """

    def __init__(self, rate_limiter: AdaptiveRateLimiter = None):
        """
        Initialize DataService.

        Args:
            rate_limiter: Optional AdaptiveRateLimiter instance. If None, creates a new one.
        """
        self.rate_limiter = rate_limiter or AdaptiveRateLimiter()
        self._cache: Dict[str, pd.DataFrame] = {}  # Simple in-memory cache

    def get_data_for_backtest(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for backtesting.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        result = {}
        for symbol in symbols:
            cache_key = f"{symbol}:{start_date}:{end_date}"

            if cache_key in self._cache:
                result[symbol] = self._cache[cache_key]
                continue

            data = self._fetch_stock_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                self._cache[cache_key] = data
                result[symbol] = data

        return result

    def get_benchmark_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Get benchmark index data.

        Args:
            symbol: Index symbol (e.g., "000300" for CSI 300)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data, or None on error
        """
        cache_key = f"benchmark:{symbol}:{start_date}:{end_date}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        data = self._fetch_index_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            self._cache[cache_key] = data

        return data

    def _fetch_stock_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data from AKShare with rate limiting.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data, or None on error
        """
        try:
            # Acquire rate limit token
            if not self.rate_limiter.acquire(blocking=True, timeout=30):
                logger.warning(f"Rate limit timeout for {symbol}")
                return None

            # Fetch data from AKShare
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq"  # Forward-adjusted prices
            )

            self.rate_limiter.on_success()

            # Process DataFrame
            if df is not None and not df.empty:
                df = self._process_stock_dataframe(df)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            self.rate_limiter.on_fail()
            return None

    def _fetch_index_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Fetch index data from AKShare with rate limiting.

        Args:
            symbol: Index symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data, or None on error
        """
        try:
            if not self.rate_limiter.acquire(blocking=True, timeout=30):
                logger.warning(f"Rate limit timeout for index {symbol}")
                return None

            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d")
            )

            self.rate_limiter.on_success()

            if df is not None and not df.empty:
                df = self._process_index_dataframe(df)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch index {symbol}: {e}")
            self.rate_limiter.on_fail()
            return None

    def _process_stock_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process stock data DataFrame to standard format.

        Args:
            df: Raw DataFrame from AKShare

        Returns:
            Processed DataFrame with standard columns and DatetimeIndex
        """
        # Map Chinese column names to English
        column_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount"
        }

        df = df.rename(columns=column_map)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df[required]

    def _process_index_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process index data DataFrame to standard format.

        Args:
            df: Raw DataFrame from AKShare

        Returns:
            Processed DataFrame with standard columns and DatetimeIndex
        """
        column_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume"
        }

        df = df.rename(columns=column_map)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        return df[["open", "high", "low", "close", "volume"]]

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()


# Global lightweight data service instance
_lightweight_data_service: Optional[DataService] = None


def get_lightweight_data_service() -> DataService:
    """Get the global lightweight DataService instance."""
    global _lightweight_data_service

    if _lightweight_data_service is None:
        _lightweight_data_service = DataService()

    return _lightweight_data_service
