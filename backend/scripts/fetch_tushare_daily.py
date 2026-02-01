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
    
    print("正在获取A股股票列表...")
    return service.get_all_stocks()


def main():
    args = parse_args()
    
    if not args.resume and not args.retry_failed and not args.start:
        print("错误: 必须指定 --start 参数或使用 --resume/--retry-failed")
        sys.exit(1)
    
    try:
        service = get_tushare_service()
    except ValueError as e:
        print(f"错误: {e}")
        print("请在 .env 文件中配置 TUSHARE_TOKEN")
        sys.exit(1)
    
    if not args.force_new and not args.resume and not args.retry_failed:
        tracker = ProgressTracker()
        progress = tracker.load()
        if progress and not tracker.is_expired():
            completed = len(progress.get('completed_symbols', []))
            total = progress.get('total_symbols', 0)
            last_update = progress.get('last_update', '')
            
            print(f"\n检测到未完成的任务 ({last_update}):")
            print(f"  - 已完成: {completed}/{total} ({completed/total*100:.1f}%)" if total > 0 else "  - 已完成: 0")
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
    
    symbols = get_symbols(args, service)
    print(f"共 {len(symbols)} 只股票待处理")
    
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
    
    stats = service.fetch_batch(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        incremental=args.incremental,
        resume=args.resume,
        force_new=args.force_new
    )
    
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
