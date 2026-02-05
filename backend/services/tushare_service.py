"""
Tushare数据服务 - 从tushare拉取A股日线数据
"""
import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Set

import tushare as ts
import pandas as pd
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from database.connection import get_session

logger = logging.getLogger(__name__)


class TushareRateLimiter:
    """Tushare API频率限制器"""
    
    def __init__(self, rate: float = 6.0):
        self.rate = rate
        self.min_interval = 1.0 / rate
        self.last_request_time = 0.0
    
    def wait(self):
        """等待直到可以发送下一个请求"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class ProgressTracker:
    """断点续传进度跟踪器"""
    
    def __init__(self, progress_file: str = None):
        self.progress_file = progress_file or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", ".tushare_fetch_progress.json"
        )
        self.data = {}
    
    def load(self) -> Optional[Dict]:
        """加载进度文件"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                return self.data
        return None
    
    def save(self):
        """保存进度"""
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def delete(self):
        """删除进度文件"""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
    
    def is_expired(self, days: int = 7) -> bool:
        """检查进度文件是否过期"""
        if not self.data:
            return True
        last_update = self.data.get("last_update")
        if not last_update:
            return True
        last_dt = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - last_dt).days > days


class TushareService:
    """Tushare数据服务"""
    
    def __init__(self):
        if not config.TUSHARE_TOKEN:
            raise ValueError("TUSHARE_TOKEN not configured")
        
        ts.set_token(config.TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        self.rate_limiter = TushareRateLimiter(config.TUSHARE_RATE_LIMIT)
        self.progress_tracker = ProgressTracker()
        self.retry_times = config.TUSHARE_RETRY_TIMES
        self.retry_delay = config.TUSHARE_RETRY_DELAY
    
    def get_all_stocks(self) -> List[str]:
        """获取所有A股股票列表"""
        self.rate_limiter.wait()
        df = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,name,list_date'
        )
        return df['ts_code'].tolist()
    
    def fetch_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """拉取单只股票的日线数据（带重试）"""
        for attempt in range(self.retry_times):
            try:
                self.rate_limiter.wait()
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                return df
            except Exception as e:
                if attempt < self.retry_times - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"{ts_code} 第{attempt+1}次失败: {e}, {delay}秒后重试")
                    time.sleep(delay)
                else:
                    logger.error(f"{ts_code} 最终失败: {e}")
                    raise
        return None

    def fetch_daily_by_date(self, trade_date: str) -> Optional[pd.DataFrame]:
        """按交易日期拉取所有股票的日线数据（带重试）"""
        for attempt in range(self.retry_times):
            try:
                self.rate_limiter.wait()
                df = self.pro.daily(trade_date=trade_date)
                return df
            except Exception as e:
                if attempt < self.retry_times - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"trade_date={trade_date} 第{attempt+1}次失败: {e}, {delay}秒后重试")
                    time.sleep(delay)
                else:
                    logger.error(f"trade_date={trade_date} 最终失败: {e}")
                    raise
        return None

    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取指定日期范围内的交易日列表"""
        self.rate_limiter.wait()
        df = self.pro.trade_cal(
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            is_open='1'
        )
        return df['cal_date'].tolist()
    
    def save_to_db(self, df: pd.DataFrame) -> int:
        """保存数据到数据库"""
        if df is None or df.empty:
            return 0

        def clean_value(val):
            """Convert NaN to None for MySQL compatibility"""
            if pd.isna(val):
                return None
            return val

        records = df.to_dict('records')
        with get_session() as session:
            for record in records:
                data = {
                    'ts_code': record['ts_code'],
                    'trade_date': datetime.strptime(record['trade_date'], '%Y%m%d').date(),
                    'open': clean_value(record.get('open')),
                    'high': clean_value(record.get('high')),
                    'low': clean_value(record.get('low')),
                    'close': clean_value(record.get('close')),
                    'pre_close': clean_value(record.get('pre_close')),
                    'change_amt': clean_value(record.get('change')),
                    'pct_chg': clean_value(record.get('pct_chg')),
                    'vol': clean_value(record.get('vol')),
                    'amount': clean_value(record.get('amount')),
                }
                
                stmt = text("""
                    INSERT INTO tushare_stock_daily 
                    (ts_code, trade_date, open, high, low, close, pre_close, 
                     change_amt, pct_chg, vol, amount)
                    VALUES (:ts_code, :trade_date, :open, :high, :low, :close, 
                            :pre_close, :change_amt, :pct_chg, :vol, :amount)
                    ON DUPLICATE KEY UPDATE
                    open=VALUES(open), high=VALUES(high), low=VALUES(low),
                    close=VALUES(close), pre_close=VALUES(pre_close),
                    change_amt=VALUES(change_amt), pct_chg=VALUES(pct_chg),
                    vol=VALUES(vol), amount=VALUES(amount),
                    updated_at=NOW()
                """)
                session.execute(stmt, data)
        
        return len(records)
    
    def get_existing_dates(self, ts_code: str, start_date: str, end_date: str) -> Set[str]:
        """获取数据库中已存在的日期"""
        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT DATE_FORMAT(trade_date, '%Y%m%d') as trade_date
                    FROM tushare_stock_daily
                    WHERE ts_code = :ts_code
                    AND trade_date BETWEEN :start_date AND :end_date
                """),
                {
                    'ts_code': ts_code,
                    'start_date': datetime.strptime(start_date, '%Y%m%d').date(),
                    'end_date': datetime.strptime(end_date, '%Y%m%d').date()
                }
            )
            return {row[0] for row in result}
    
    def fetch_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        incremental: bool = False,
        resume: bool = False,
        force_new: bool = False
    ) -> Dict:
        """批量拉取数据"""
        progress = None
        if resume and not force_new:
            progress = self.progress_tracker.load()
            if progress and not self.progress_tracker.is_expired():
                completed = set(progress.get('completed_symbols', []))
                symbols = [s for s in symbols if s not in completed]
                logger.info(f"从断点继续，剩余 {len(symbols)} 只股票")
        
        if not progress or force_new:
            self.progress_tracker.data = {
                'task_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'start_date': start_date,
                'end_date': end_date,
                'total_symbols': len(symbols),
                'completed_symbols': [],
                'failed_symbols': {},
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        stats = {'success': 0, 'failed': 0, 'skipped': 0, 'records': 0}
        total = len(symbols)
        start_time = time.time()
        
        for i, ts_code in enumerate(symbols):
            try:
                if incremental:
                    existing = self.get_existing_dates(ts_code, start_date, end_date)
                    if existing:
                        stats['skipped'] += 1
                        self.progress_tracker.data['completed_symbols'].append(ts_code)
                        continue
                
                df = self.fetch_daily(ts_code, start_date, end_date)
                
                if df is not None and not df.empty:
                    count = self.save_to_db(df)
                    stats['records'] += count
                    stats['success'] += 1
                else:
                    stats['skipped'] += 1
                
                self.progress_tracker.data['completed_symbols'].append(ts_code)
                
            except Exception as e:
                stats['failed'] += 1
                self.progress_tracker.data['failed_symbols'][ts_code] = {
                    'error': str(e),
                    'attempts': self.retry_times,
                    'last_attempt': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            self.progress_tracker.data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if (i + 1) % 10 == 0:
                self.progress_tracker.save()
                
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - i - 1) / speed if speed > 0 else 0
                pct = (i + 1) / total * 100
                
                print(f"\r[{'=' * int(pct/5):20s}] {pct:.1f}% | {i+1}/{total} | {ts_code} | 剩余: {remaining/60:.1f}分钟", end='', flush=True)
        
        print()
        
        if stats['failed'] == 0:
            self.progress_tracker.delete()
        else:
            self.progress_tracker.save()
            failed_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                f"failed_symbols_{datetime.now().strftime('%Y%m%d')}.txt"
            )
            os.makedirs(os.path.dirname(failed_file), exist_ok=True)
            with open(failed_file, 'w') as f:
                for code in self.progress_tracker.data['failed_symbols']:
                    f.write(f"{code}\n")
            print(f"\n失败股票已保存到: {failed_file}")

        return stats

    def fetch_batch_by_date(
        self,
        start_date: str,
        end_date: str,
        resume: bool = False,
        force_new: bool = False
    ) -> Dict:
        """按交易日期批量拉取数据（每次拉取一个交易日的所有股票数据）"""
        trade_dates = self.get_trade_dates(start_date, end_date)

        progress = None
        if resume and not force_new:
            progress = self.progress_tracker.load()
            if progress and not self.progress_tracker.is_expired():
                completed = set(progress.get('completed_dates', []))
                trade_dates = [d for d in trade_dates if d not in completed]
                logger.info(f"从断点继续，剩余 {len(trade_dates)} 个交易日")

        if not progress or force_new:
            self.progress_tracker.data = {
                'task_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'mode': 'by_date',
                'start_date': start_date,
                'end_date': end_date,
                'total_dates': len(trade_dates),
                'completed_dates': [],
                'failed_dates': {},
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        stats = {'success': 0, 'failed': 0, 'skipped': 0, 'records': 0}
        total = len(trade_dates)
        start_time = time.time()

        for i, trade_date in enumerate(trade_dates):
            try:
                df = self.fetch_daily_by_date(trade_date)

                if df is not None and not df.empty:
                    count = self.save_to_db(df)
                    stats['records'] += count
                    stats['success'] += 1
                else:
                    stats['skipped'] += 1

                self.progress_tracker.data['completed_dates'].append(trade_date)

            except Exception as e:
                stats['failed'] += 1
                self.progress_tracker.data['failed_dates'][trade_date] = {
                    'error': str(e),
                    'attempts': self.retry_times,
                    'last_attempt': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

            self.progress_tracker.data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if (i + 1) % 5 == 0:
                self.progress_tracker.save()

                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - i - 1) / speed if speed > 0 else 0
                pct = (i + 1) / total * 100

                print(f"\r[{'=' * int(pct/5):20s}] {pct:.1f}% | {i+1}/{total} | {trade_date} | 剩余: {remaining/60:.1f}分钟", end='', flush=True)

        print()

        if stats['failed'] == 0:
            self.progress_tracker.delete()
        else:
            self.progress_tracker.save()
            failed_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                f"failed_dates_{datetime.now().strftime('%Y%m%d')}.txt"
            )
            os.makedirs(os.path.dirname(failed_file), exist_ok=True)
            with open(failed_file, 'w') as f:
                for date in self.progress_tracker.data['failed_dates']:
                    f.write(f"{date}\n")
            print(f"\n失败日期已保存到: {failed_file}")

        return stats

    def fetch_stock_basic(self) -> Optional[pd.DataFrame]:
        """获取所有A股股票基础信息"""
        try:
            self.rate_limiter.wait()
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date,list_status'
            )
            return df
        except Exception as e:
            logger.error(f"获取股票基础信息失败: {e}")
            return None

    def save_stock_basic_to_db(self) -> int:
        """拉取并保存股票基础信息到数据库"""
        df = self.fetch_stock_basic()
        if df is None or df.empty:
            return 0

        records = df.to_dict('records')
        with get_session() as session:
            for record in records:
                list_date = record.get('list_date')
                if list_date and pd.notna(list_date):
                    list_date = datetime.strptime(str(list_date), '%Y%m%d').date()
                else:
                    list_date = None

                data = {
                    'ts_code': record['ts_code'],
                    'symbol': record.get('symbol'),
                    'name': record.get('name'),
                    'area': record.get('area'),
                    'industry': record.get('industry'),
                    'market': record.get('market'),
                    'list_date': list_date,
                    'list_status': record.get('list_status'),
                }

                stmt = text("""
                    INSERT INTO tushare_stock_basic
                    (ts_code, symbol, name, area, industry, market, list_date, list_status)
                    VALUES (:ts_code, :symbol, :name, :area, :industry, :market, :list_date, :list_status)
                    ON DUPLICATE KEY UPDATE
                    symbol=VALUES(symbol), name=VALUES(name), area=VALUES(area),
                    industry=VALUES(industry), market=VALUES(market),
                    list_date=VALUES(list_date), list_status=VALUES(list_status),
                    updated_at=NOW()
                """)
                session.execute(stmt, data)

        logger.info(f"保存了 {len(records)} 条股票基础信息")
        return len(records)


_tushare_service = None

def get_tushare_service() -> TushareService:
    """获取全局TushareService实例"""
    global _tushare_service
    if _tushare_service is None:
        _tushare_service = TushareService()
    return _tushare_service
