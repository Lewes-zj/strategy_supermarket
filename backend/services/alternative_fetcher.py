"""
Alternative stock data fetcher using multiple data sources.
Falls back between different sources when one fails.
"""
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class AlternativeDataFetcher:
    """
    Fetch stock data from alternative sources when AkShare/eastmoney fails.

    Sources:
    1. Sina Finance (stock_zh_a_daily)
    2. NetEase Finance (stock_zh_a_hist_163)
    3. Tenxun Finance
    """

    def __init__(self):
        self.source_priority = ['sina', '163', 'tx']

    def fetch_sina(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Sina Finance using stock_zh_a_daily.

        Sina's API is more stable and doesn't have the anti-scraping issues of eastmoney.
        Returns all historical data which can be filtered by date range.
        """
        try:
            import akshare as ak

            # Convert symbol to Sina format (6-digit with sh/sz prefix)
            original_symbol = symbol
            if len(symbol) == 6:
                if symbol.startswith('6'):
                    symbol = f'sh{symbol}'
                elif symbol.startswith('0') or symbol.startswith('3'):
                    symbol = f'sz{symbol}'
                elif symbol.startswith('8') or symbol.startswith('4'):
                    symbol = f'bj{symbol}'

            logger.info(f"Trying Sina Finance for {original_symbol} (as {symbol})...")

            # Use Sina's historical data endpoint
            df = ak.stock_zh_a_daily(symbol=symbol)

            if df is None or df.empty:
                logger.debug(f"No data returned from Sina for {original_symbol}")
                return None

            # The data comes with 'date' column, not as index
            # Columns: ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', ...]
            df['date'] = pd.to_datetime(df['date'])

            # Filter by date range
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            end_dt = pd.to_datetime(end_date, format='%Y%m%d')
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

            if df.empty:
                logger.debug(f"No data in date range for {original_symbol}")
                return None

            # Select required columns - keep 'date' as column (not index) for database
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()

            logger.info(f"Successfully fetched {len(df)} records from Sina for {original_symbol}")
            return df

        except Exception as e:
            logger.debug(f"Sina Finance failed for {symbol}: {e}")
            return None

    def fetch_163(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from NetEase (163) Finance.

        163 Finance is generally more stable than eastmoney.
        """
        try:
            import akshare as ak

            logger.info(f"Trying NetEase Finance for {symbol}...")

            # Convert symbol format for 163
            if len(symbol) == 6:
                if symbol.startswith('6'):
                    symbol_163 = f'0{symbol}'
                else:
                    symbol_163 = f'1{symbol}'
            else:
                symbol_163 = symbol

            df = ak.stock_zh_a_hist_163(
                symbol=symbol_163,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                return None

            # Rename columns
            column_map = {
                '日期': 'date',
                '开盘价': 'open',
                '最高价': 'high',
                '最低价': 'low',
                '收盘价': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            }

            # Check actual columns in dataframe
            actual_columns = df.columns.tolist()
            logger.debug(f"163 columns: {actual_columns}")

            df = df.rename(columns=column_map)

            # Ensure we have the required columns
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            df.set_index('date', inplace=True)

            return df

        except Exception as e:
            logger.debug(f"NetEase Finance failed for {symbol}: {e}")
            return None

    def fetch_jp_stock(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Use jpstock library as alternative (if installed).
        """
        try:
            import jpstock as jp

            logger.info(f"Trying jpstock for {symbol}...")

            # Convert to jpstock format
            code = symbol
            if symbol.startswith('6'):
                market = 'sh'
            else:
                market = 'sz'

            df = jp.Stock(code, market).get()

            if df is None or df.empty:
                return None

            # Filter and format
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            end_dt = pd.to_datetime(end_date, format='%Y%m%d')
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['amount'] = df['volume'] * df['close']
            df.set_index('date', inplace=True)

            return df

        except ImportError:
            logger.debug("jpstock not installed")
            return None
        except Exception as e:
            logger.debug(f"jpstock failed for {symbol}: {e}")
            return None

    def fetch_with_fallback(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Try multiple data sources until one succeeds.

        Returns:
            DataFrame with stock data or None if all sources fail
        """
        # Try Sina first
        df = self.fetch_sina(symbol, start_date, end_date)
        if df is not None and not df.empty:
            logger.info(f"Successfully fetched {symbol} from Sina Finance")
            return df

        time.sleep(1)

        # Try NetEase
        df = self.fetch_163(symbol, start_date, end_date)
        if df is not None and not df.empty:
            logger.info(f"Successfully fetched {symbol} from NetEase Finance")
            return df

        time.sleep(1)

        # Try jpstock
        df = self.fetch_jp_stock(symbol, start_date, end_date)
        if df is not None and not df.empty:
            logger.info(f"Successfully fetched {symbol} from jpstock")
            return df

        logger.error(f"All data sources failed for {symbol}")
        return None


# Global instance
_fetcher = None


def get_alternative_fetcher() -> AlternativeDataFetcher:
    """Get the global alternative fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = AlternativeDataFetcher()
    return _fetcher
