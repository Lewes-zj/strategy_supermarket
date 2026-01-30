import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.data_service import get_data_service
from database.repository import StockDataRepository

# Global data service reference
_data_service = None


def _get_data_service():
    """Lazy initialization of data service."""
    global _data_service
    if _data_service is None:
        _data_service = get_data_service()
    return _data_service


def fetch_stock_data(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """
    Fetch A-share stock data using AkShare with rate limiting and caching.
    symbol: e.g. "000001"
    start_date: "YYYYMMDD"
    end_date: "YYYYMMDD"
    adjust: "qfq" (forward), "hfq" (backward), or "" (none)
    """
    try:
        data_service = _get_data_service()

        # Try to get from cache first
        start_dt = datetime.strptime(start_date, "%Y%m%d").date()
        end_dt = datetime.strptime(end_date, "%Y%m%d").date() if end_date else datetime.now().date()

        # Get cached data
        df = data_service.get_cached_data([symbol], start_dt, end_dt)

        if df.empty:
            # Fallback to direct AkShare fetch with rate limiting
            df = _fetch_from_akshare(symbol, start_date, end_date, adjust)

        return df

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def _fetch_from_akshare(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """Direct fetch from AkShare with rate limiting."""
    try:
        data_service = _get_data_service()
        df = data_service.fetch_stock_data(symbol, start_date, end_date, adjust)

        if df is not None and not df.empty:
            # Save to database
            StockDataRepository.save_stock_data(symbol, df)
            df['symbol'] = symbol
            return df.set_index('date') if 'date' in df.columns else df

        return pd.DataFrame()

    except Exception as e:
        print(f"Error in direct AkShare fetch for {symbol}: {e}")
        return pd.DataFrame()


def fetch_multiple_symbols(
    symbols: List[str],
    start_date: str,
    end_date: str = None,
    adjust: str = "qfq"
) -> pd.DataFrame:
    """
    Fetch data for multiple symbols efficiently.

    Args:
        symbols: List of stock symbols
        start_date: Start date "YYYYMMDD"
        end_date: End date "YYYYMMDD"
        adjust: Adjustment type

    Returns:
        DataFrame with multi-index or symbol column
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    start_dt = datetime.strptime(start_date, "%Y%m%d").date()
    end_dt = datetime.strptime(end_date, "%Y%m%d").date()

    data_service = _get_data_service()

    # Get from cache first
    df = data_service.get_cached_data(symbols, start_dt, end_dt)

    if df.empty:
        # Fetch all symbols
        all_data = []
        for symbol in symbols:
            symbol_df = fetch_stock_data(symbol, start_date, end_date, adjust)
            if not symbol_df.empty:
                all_data.append(symbol_df)

        if all_data:
            df = pd.concat(all_data)

    return df


def get_stock_pool() -> List[str]:
    """Get the active stock pool symbols."""
    try:
        data_service = _get_data_service()
        return data_service.stock_pool_repo.get_active_symbols("CSI300")
    except Exception as e:
        print(f"Error getting stock pool: {e}")
        return []


def get_stock_pool_details() -> List[dict]:
    """Get detailed stock pool information."""
    try:
        data_service = _get_data_service()
        return data_service.stock_pool_repo.get_stock_pool()
    except Exception as e:
        print(f"Error getting stock pool details: {e}")
        return []


def generate_mock_data(start_date: str = "20230101", days: int = 750, symbol: str = "MOCK01") -> pd.DataFrame:
    """
    Generate realistic-looking random stock data.

    Args:
        start_date: Start date in YYYYMMDD format (default: 20230101)
        days: Number of days to generate (default: 750 = ~3 years)
        symbol: Stock symbol
    """
    dates = pd.date_range(start=start_date, periods=days, freq="B")

    # Use symbol as seed for reproducibility per symbol
    seed = hash(symbol) % (2**31)
    np.random.seed(seed)

    # Geometric Brownian Motion
    dt = 1/252
    mu = 0.08  # 8% annual drift (more realistic)
    sigma = 0.25  # 25% annual volatility

    current_price = 100.0
    data = []

    # Pre-generate all random shocks for speed
    n = len(dates)
    shocks = np.random.normal(0, np.sqrt(dt), n)
    high_shocks = np.random.normal(0, 0.5, n)
    low_shocks = np.random.normal(0, 0.5, n)
    open_shocks = np.random.normal(0, 0.25, n)

    for i, date in enumerate(dates):
        # Simple GBM step
        shock = shocks[i]
        change = current_price * (mu * dt + sigma * shock)
        close_price = current_price + change

        # Generate OHLC
        daily_vol = current_price * sigma * np.sqrt(dt)
        high_price = close_price + abs(daily_vol * high_shocks[i] / 2)
        low_price = close_price - abs(daily_vol * low_shocks[i] / 2)
        open_price = (high_price + low_price) / 2 + daily_vol * open_shocks[i] / 4

        # Ensure logic
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        volume = int(np.random.normal(1000000, 200000))

        data.append({
            "date": date,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume,
            "symbol": symbol
        })
        current_price = close_price

    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    return df


def init_stock_database() -> bool:
    """
    Initialize the stock database with CSI 300 stock pool and historical data.

    Returns:
        True if successful
    """
    try:
        data_service = _get_data_service()

        # Initialize stock pool
        print("Initializing CSI 300 stock pool...")
        count = data_service.init_stock_pool()
        print(f"Added {count} symbols to stock pool")

        # Update historical data (last 2 years)
        print("Updating historical data...")
        stats = data_service.update_stock_data(days_back=365*2)
        print(f"Update complete: {stats}")

        return True

    except Exception as e:
        print(f"Error initializing stock database: {e}")
        return False
