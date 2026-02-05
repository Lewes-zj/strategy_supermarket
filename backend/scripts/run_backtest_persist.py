#!/usr/bin/env python
"""
回测脚本：运行策略回测并将结果持久化到数据库

用法:
    python scripts/run_backtest_persist.py --strategy alpha_trend
    python scripts/run_backtest_persist.py --strategy dragon_leader
    python scripts/run_backtest_persist.py --strategy dragon_leader --days 90
    python scripts/run_backtest_persist.py --all  # 运行所有策略
"""
import argparse
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_all_strategy_ids() -> List[str]:
    """获取所有已注册的策略ID"""
    from strategies.registry import STRATEGY_REGISTRY
    return list(STRATEGY_REGISTRY.keys())


def run_and_persist(strategy_id: str, backtest_days: int = None) -> bool:
    """运行回测并持久化结果

    Args:
        strategy_id: 策略ID
        backtest_days: 回测天数，如果指定则覆盖策略默认值
    """
    from database.connection import get_session
    from database.repository import StockPoolRepository
    from services.backtest_persistence import BacktestPersistenceService

    logger.info(f"开始回测策略: {strategy_id}")

    try:
        # 导入 main 模块中的 run_backtest 函数
        from main import run_backtest

        # 运行回测
        result = run_backtest(strategy_id, use_real_data=True, backtest_days=backtest_days)

        if not result:
            logger.warning(f"策略 {strategy_id} 回测结果为空")
            return False

        # 获取行业映射
        stock_pool_repo = StockPoolRepository()
        stock_pool = stock_pool_repo.get_stock_pool()
        sector_map = {s['symbol']: s.get('sector', '其他') for s in stock_pool}

        # 构建 BacktestResult 对象
        from engine.models import BacktestResult, PerformanceMetrics
        from dataclasses import fields

        # Handle both fresh backtest (has backtester) and cached result (has trades list)
        backtester = result.get("backtester")
        if backtester:
            # Fresh backtest result
            trades = backtester.trades
            positions = backtester.portfolio.positions if hasattr(backtester, 'portfolio') else {}
        else:
            # Cached result - trades is a list of dicts, no positions available
            trades = result.get("trades", [])
            positions = {}
            logger.info(f"Using cached result for {strategy_id}")

        # Filter metrics to only include valid PerformanceMetrics fields
        metrics_dict = result["metrics"]
        if isinstance(metrics_dict, dict):
            valid_fields = {f.name for f in fields(PerformanceMetrics)}
            filtered_metrics = {k: v for k, v in metrics_dict.items() if k in valid_fields}
            metrics = PerformanceMetrics(**filtered_metrics)
        else:
            metrics = metrics_dict

        backtest_result = BacktestResult(
            equity_curve=result["equity"],
            trades=trades,
            metrics=metrics,
            positions=positions
        )

        # 持久化到数据库
        with get_session() as db:
            persistence_service = BacktestPersistenceService(db)
            persistence_service.save_backtest_result(
                strategy_id=strategy_id,
                result=backtest_result,
                sector_map=sector_map
            )

        logger.info(f"策略 {strategy_id} 回测完成，已写入数据库")
        logger.info(f"  - 交易记录: {len(trades)} 条")
        logger.info(f"  - 权益曲线: {len(result['equity'])} 天")

        return True

    except Exception as e:
        logger.error(f"策略 {strategy_id} 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='运行策略回测并持久化结果')
    parser.add_argument('--strategy', '-s', type=str, help='策略ID (如 alpha_trend, dragon_leader)')
    parser.add_argument('--days', '-d', type=int, help='回测天数，覆盖策略默认值')
    parser.add_argument('--all', '-a', action='store_true', help='运行所有策略')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有可用策略')

    args = parser.parse_args()

    # 列出所有策略
    if args.list:
        strategies = get_all_strategy_ids()
        print("可用策略:")
        for s in strategies:
            print(f"  - {s}")
        return

    # 确定要运行的策略
    if args.all:
        strategy_ids = get_all_strategy_ids()
    elif args.strategy:
        strategy_ids = [args.strategy]
    else:
        parser.print_help()
        return

    # 初始化数据库表
    logger.info("初始化数据库表...")
    from database.connection import init_db
    init_db()

    # 运行回测
    success_count = 0
    fail_count = 0

    for strategy_id in strategy_ids:
        if run_and_persist(strategy_id, backtest_days=args.days):
            success_count += 1
        else:
            fail_count += 1

    # 输出结果
    print("\n" + "=" * 50)
    print(f"回测完成: 成功 {success_count}, 失败 {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
