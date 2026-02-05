"""
Tushare涨跌停价格数据服务 - 从tushare拉取A股每日涨跌停价格
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
from services.tushare_service import TushareRateLimiter

logger = logging.getLogger(__name__)


class LimitProgressTracker:
    """涨跌停数据断点续传进度跟踪器"""

    def __init__(self, progress_file: str = None):
        self.progress_file = progress_file or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", ".tushare_limit_progress.json"
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


class TushareLimitService:
    """Tushare涨跌停价格数据服务"""

    def __init__(self):
        if not config.TUSHARE_TOKEN:
            raise ValueError("TUSHARE_TOKEN not configured")

        ts.set_token(config.TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        self.rate_limiter = TushareRateLimiter(config.TUSHARE_RATE_LIMIT)
        self.progress_tracker = LimitProgressTracker()
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

    def fetch_limit(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """拉取单只股票的涨跌停价格数据（带重试）"""
        for attempt in range(self.retry_times):
            try:
                self.rate_limiter.wait()
                df = self.pro.stk_limit(
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

    def fetch_by_trade_date(self, trade_date: str) -> Optional[pd.DataFrame]:
        """按交易日期拉取当天所有股票的涨跌停价格数据（带重试）"""
        for attempt in range(self.retry_times):
            try:
                self.rate_limiter.wait()
                df = self.pro.stk_limit(trade_date=trade_date)
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

    def fetch_by_trade_dates(
        self,
        start_date: str,
        end_date: str
    ) -> Dict:
        """按日期范围逐日拉取所有股票的涨跌停价格数据"""
        from datetime import datetime, timedelta

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        stats = {'success': 0, 'failed': 0, 'records': 0, 'failed_dates': []}
        current_dt = start_dt
        total_days = (end_dt - start_dt).days + 1
        processed = 0

        while current_dt <= end_dt:
            trade_date = current_dt.strftime('%Y%m%d')
            processed += 1

            try:
                df = self.fetch_by_trade_date(trade_date)

                if df is not None and not df.empty:
                    count = self.save_to_db(df)
                    stats['records'] += count
                    stats['success'] += 1
                    print(f"\r[{processed}/{total_days}] {trade_date}: {count} 条记录", end='', flush=True)
                else:
                    # 非交易日或无数据
                    print(f"\r[{processed}/{total_days}] {trade_date}: 无数据(非交易日)", end='', flush=True)

            except Exception as e:
                stats['failed'] += 1
                stats['failed_dates'].append(trade_date)
                logger.error(f"{trade_date} 拉取失败: {e}")

            current_dt += timedelta(days=1)

        print()
        return stats

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
                    'pre_close': clean_value(record.get('pre_close')),
                    'up_limit': clean_value(record.get('up_limit')),
                    'down_limit': clean_value(record.get('down_limit')),
                }

                stmt = text("""
                    INSERT INTO tushare_stock_limit
                    (ts_code, trade_date, pre_close, up_limit, down_limit)
                    VALUES (:ts_code, :trade_date, :pre_close, :up_limit, :down_limit)
                    ON DUPLICATE KEY UPDATE
                    pre_close=VALUES(pre_close), up_limit=VALUES(up_limit),
                    down_limit=VALUES(down_limit), updated_at=NOW()
                """)
                session.execute(stmt, data)

        return len(records)

    def get_existing_dates(self, ts_code: str, start_date: str, end_date: str) -> Set[str]:
        """获取数据库中已存在的日期"""
        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT DATE_FORMAT(trade_date, '%Y%m%d') as trade_date
                    FROM tushare_stock_limit
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

                df = self.fetch_limit(ts_code, start_date, end_date)

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
                f"failed_limit_symbols_{datetime.now().strftime('%Y%m%d')}.txt"
            )
            os.makedirs(os.path.dirname(failed_file), exist_ok=True)
            with open(failed_file, 'w') as f:
                for code in self.progress_tracker.data['failed_symbols']:
                    f.write(f"{code}\n")
            print(f"\n失败股票已保存到: {failed_file}")

        return stats


_tushare_limit_service = None

def get_tushare_limit_service() -> TushareLimitService:
    """获取全局TushareLimitService实例"""
    global _tushare_limit_service
    if _tushare_limit_service is None:
        _tushare_limit_service = TushareLimitService()
    return _tushare_limit_service
