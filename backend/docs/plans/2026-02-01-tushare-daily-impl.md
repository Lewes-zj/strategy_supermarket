# Tushare日线数据拉取 - 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现从tushare拉取A股历史日线数据的脚本，支持增量更新、断点续传、失败重试

**Architecture:** 分层架构 - 配置层(config.py) → 服务层(tushare_service.py) → 脚本层(fetch_tushare_daily.py) → 数据层(models.py)

**Tech Stack:** Python, tushare, SQLAlchemy, MySQL分区表, argparse

---

## Task 1: 添加配置项

**Files:**
- Modify: `backend/config.py`

**Step 1: 添加tushare配置项**

在config.py的Config类中添加：

```python
# Tushare Settings
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
TUSHARE_RATE_LIMIT = float(os.getenv("TUSHARE_RATE_LIMIT", "6.0"))
TUSHARE_RETRY_TIMES = int(os.getenv("TUSHARE_RETRY_TIMES", "3"))
TUSHARE_RETRY_DELAY = float(os.getenv("TUSHARE_RETRY_DELAY", "1.0"))
```

**Step 2: 提交**

```bash
git add backend/config.py
git commit -m "feat: add tushare configuration"
```

---

## Task 2: 创建数据库模型

**Files:**
- Modify: `backend/database/models.py`

**Step 1: 添加TushareStockDaily模型**

```python
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
```

**Step 2: 提交**

```bash
git add backend/database/models.py
git commit -m "feat: add TushareStockDaily model"
```

---

## Task 3: 创建分区表SQL脚本

**Files:**
- Create: `backend/scripts/create_tushare_partition_table.sql`

**Step 1: 创建SQL文件**

```sql
-- 创建tushare日线数据分区表
-- 注意：SQLAlchemy不支持自动创建分区表，需要手动执行此SQL

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
```

**Step 2: 提交**

```bash
git add backend/scripts/create_tushare_partition_table.sql
git commit -m "feat: add tushare partition table SQL script"
```

---

## Task 4: 创建Tushare服务类

**Files:**
- Create: `backend/services/tushare_service.py`

**Step 1: 创建服务类框架**

```python
"""
Tushare数据服务 - 从tushare拉取A股日线数据
"""
import os
import sys
import time
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Set
from pathlib import Path

import tushare as ts
import pandas as pd
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from database.connection import get_session, get_engine
from database.models import TushareStockDaily

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
    
    def save_to_db(self, df: pd.DataFrame) -> int:
        """保存数据到数据库"""
        if df is None or df.empty:
            return 0
        
        records = df.to_dict('records')
        with get_session() as session:
            for record in records:
                # 转换字段名
                data = {
                    'ts_code': record['ts_code'],
                    'trade_date': datetime.strptime(record['trade_date'], '%Y%m%d').date(),
                    'open': record.get('open'),
                    'high': record.get('high'),
                    'low': record.get('low'),
                    'close': record.get('close'),
                    'pre_close': record.get('pre_close'),
                    'change_amt': record.get('change'),
                    'pct_chg': record.get('pct_chg'),
                    'vol': record.get('vol'),
                    'amount': record.get('amount'),
                }
                
                # UPSERT
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
        # 检查断点续传
        progress = None
        if resume and not force_new:
            progress = self.progress_tracker.load()
            if progress and not self.progress_tracker.is_expired():
                completed = set(progress.get('completed_symbols', []))
                symbols = [s for s in symbols if s not in completed]
                logger.info(f"从断点继续，剩余 {len(symbols)} 只股票")
        
        # 初始化进度
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
                # 增量更新检查
                if incremental:
                    existing = self.get_existing_dates(ts_code, start_date, end_date)
                    if existing:
                        # 简化处理：如果有数据就跳过
                        stats['skipped'] += 1
                        self.progress_tracker.data['completed_symbols'].append(ts_code)
                        continue
                
                # 拉取数据
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
            
            # 更新进度
            self.progress_tracker.data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 每10只股票保存一次进度
            if (i + 1) % 10 == 0:
                self.progress_tracker.save()
                
                # 显示进度
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - i - 1) / speed if speed > 0 else 0
                pct = (i + 1) / total * 100
                
                print(f"\r[{'=' * int(pct/5):20s}] {pct:.1f}% | {i+1}/{total} | {ts_code} | 剩余: {remaining/60:.1f}分钟", end='', flush=True)
        
        print()  # 换行
        
        # 任务完成，删除进度文件
        if stats['failed'] == 0:
            self.progress_tracker.delete()
        else:
            self.progress_tracker.save()
            # 保存失败列表
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


# 全局服务实例
_tushare_service = None

def get_tushare_service() -> TushareService:
    """获取全局TushareService实例"""
    global _tushare_service
    if _tushare_service is None:
        _tushare_service = TushareService()
    return _tushare_service
```

**Step 2: 提交**

```bash
git add backend/services/tushare_service.py
git commit -m "feat: add TushareService with rate limiting and progress tracking"
```

---

## Task 5: 创建命令行脚本

**Files:**
- Create: `backend/scripts/fetch_tushare_daily.py`

**Step 1: 创建脚本**

```python
#!/usr/bin/env python
"""
从Tushare拉取A股历史日线数据

用法:
    python scripts/fetch_tushare_daily.py --start 20200101 --end 20241231
    python scripts/fetch_tushare_daily.py --start 20200101 --incremental
    python scripts/fetch_tushare_daily.py --symbols 000001.SZ,600000.SH --start 20200101
    python scripts/fetch_tushare_daily.py --symbols-file stocks.txt --start 20200101
    python scripts/fetch_tushare_daily.py --resume
    python scripts/fetch_tushare_daily.py --retry-failed
"""
import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.tushare_service import get_tushare_service, ProgressTracker


def parse_args():
    parser = argparse.ArgumentParser(description='从Tushare拉取A股历史日线数据')
    
    parser.add_argument('--start', type=str, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y%m%d'),
                        help='结束日期 (YYYYMMDD), 默认今天')
    parser.add_argument('--symbols', type=str, help='股票代码列表，逗号分隔')
    parser.add_argument('--symbols-file', type=str, help='股票代码文件路径')
    parser.add_argument('--incremental', action='store_true', help='增量更新模式')
    parser.add_argument('--resume', action='store_true', help='继续上次未完成的任务')
    parser.add_argument('--force-new', action='store_true', help='强制重新开始')
    parser.add_argument('--retry-failed', action='store_true', help='重新拉取上次失败的股票')
    
    return parser.parse_args()


def get_symbols(args, service) -> list:
    """获取要拉取的股票列表"""
    if args.symbols:
        return args.symbols.split(',')
    
    if args.symbols_file:
        with open(args.symbols_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    if args.retry_failed:
        tracker = ProgressTracker()
        progress = tracker.load()
        if progress and progress.get('failed_symbols'):
            return list(progress['failed_symbols'].keys())
        print("没有找到失败的股票记录")
        sys.exit(1)
    
    # 默认获取所有A股
    print("正在获取A股股票列表...")
    return service.get_all_stocks()


def main():
    args = parse_args()
    
    # 参数验证
    if not args.resume and not args.retry_failed and not args.start:
        print("错误: 必须指定 --start 参数或使用 --resume/--retry-failed")
        sys.exit(1)
    
    try:
        service = get_tushare_service()
    except ValueError as e:
        print(f"错误: {e}")
        print("请在 .env 文件中配置 TUSHARE_TOKEN")
        sys.exit(1)
    
    # 检查是否有未完成的任务
    if not args.force_new and not args.resume and not args.retry_failed:
        tracker = ProgressTracker()
        progress = tracker.load()
        if progress and not tracker.is_expired():
            completed = len(progress.get('completed_symbols', []))
            total = progress.get('total_symbols', 0)
            last_update = progress.get('last_update', '')
            
            print(f"\n检测到未完成的任务 ({last_update}):")
            print(f"  - 已完成: {completed}/{total} ({completed/total*100:.1f}%)")
            print(f"  - 时间范围: {progress.get('start_date')} - {progress.get('end_date')}")
            print("\n请选择:")
            print("  [1] 继续上次任务 (--resume)")
            print("  [2] 放弃并重新开始 (--force-new)")
            print("  [3] 退出")
            
            choice = input("\n> ").strip()
            if choice == '1':
                args.resume = True
                args.start = progress.get('start_date')
                args.end = progress.get('end_date')
            elif choice == '2':
                args.force_new = True
            else:
                print("已退出")
                sys.exit(0)
    
    # 获取股票列表
    symbols = get_symbols(args, service)
    print(f"共 {len(symbols)} 只股票待处理")
    
    # 获取时间范围
    start_date = args.start
    end_date = args.end
    
    if args.resume:
        tracker = ProgressTracker()
        progress = tracker.load()
        if progress:
            start_date = progress.get('start_date', start_date)
            end_date = progress.get('end_date', end_date)
    
    print(f"时间范围: {start_date} - {end_date}")
    print(f"模式: {'增量更新' if args.incremental else '全量拉取'}")
    print("-" * 60)
    
    # 开始拉取
    stats = service.fetch_batch(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        incremental=args.incremental,
        resume=args.resume,
        force_new=args.force_new
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("拉取完成!")
    print(f"  - 成功: {stats['success']} 只")
    print(f"  - 失败: {stats['failed']} 只")
    print(f"  - 跳过: {stats['skipped']} 只")
    print(f"  - 记录数: {stats['records']} 条")
    print("=" * 60)
    
    if stats['failed'] > 0:
        print(f"\n重新拉取失败股票命令:")
        print(f"  python scripts/fetch_tushare_daily.py --retry-failed --start {start_date}")


if __name__ == '__main__':
    main()
```

**Step 2: 提交**

```bash
git add backend/scripts/fetch_tushare_daily.py
git commit -m "feat: add fetch_tushare_daily.py CLI script"
```

---

## Task 6: 添加tushare依赖

**Files:**
- Modify: `backend/requirements.txt`

**Step 1: 添加依赖**

在requirements.txt末尾添加:
```
tushare>=1.2.89
```

**Step 2: 提交**

```bash
git add backend/requirements.txt
git commit -m "feat: add tushare dependency"
```

---

## Task 7: 创建data目录

**Files:**
- Create: `backend/data/.gitkeep`

**Step 1: 创建目录和占位文件**

```bash
mkdir -p backend/data
touch backend/data/.gitkeep
echo ".tushare_fetch_progress.json" >> backend/data/.gitignore
echo "failed_symbols_*.txt" >> backend/data/.gitignore
```

**Step 2: 提交**

```bash
git add backend/data/.gitkeep backend/data/.gitignore
git commit -m "feat: add data directory for tushare progress files"
```

---

## 执行顺序

1. Task 1: 添加配置项
2. Task 2: 创建数据库模型
3. Task 3: 创建分区表SQL脚本
4. Task 4: 创建Tushare服务类
5. Task 5: 创建命令行脚本
6. Task 6: 添加tushare依赖
7. Task 7: 创建data目录

## 使用说明

1. 在 `.env` 文件中配置 `TUSHARE_TOKEN=你的token`
2. 执行SQL脚本创建分区表: `mysql -u root -p strategy_market < backend/scripts/create_tushare_partition_table.sql`
3. 安装依赖: `pip install tushare`
4. 运行脚本: `python backend/scripts/fetch_tushare_daily.py --start 20150101 --end 20260201`
