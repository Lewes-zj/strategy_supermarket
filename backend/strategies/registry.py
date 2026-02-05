"""
Strategy Registry for Strategy Supermarket.
Central registry of all available strategies with metadata.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import all strategies
from strategies.alpha_trend import AlphaTrendStrategy
from strategies.mean_reversion import MeanReversionStrategy, MultiStockMeanReversion
from strategies.momentum import MomentumStrategy, DualMomentumStrategy
from strategies.sector_rotation import (
    SectorRotationStrategy,
    RelativeStrengthSectorRotation,
    MomentumSectorRotation
)
from strategies.dragon_strategy import DragonLeaderStrategy


class StrategyInfo:
    """Information about a registered strategy."""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        description: str,
        strategy_class,
        tags: List[str],
        default_symbols: List[str],
        parameters: Dict[str, Any] = None,
        is_active: bool = True
    ):
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.strategy_class = strategy_class
        self.tags = tags
        self.default_symbols = default_symbols
        self.parameters = parameters or {}
        self.is_active = is_active

    def create_instance(self, **kwargs):
        """Create a new instance of this strategy."""
        params = {**self.parameters, **kwargs}
        return self.strategy_class(**params)


# Strategy Registry
STRATEGY_REGISTRY: Dict[str, StrategyInfo] = {
    "alpha_trend": StrategyInfo(
        strategy_id="alpha_trend",
        name="Alpha趋势策略",
        description="基于双均线交叉的趋势跟踪策略，使用10日和30日移动平均线捕捉趋势机会。",
        strategy_class=AlphaTrendStrategy,
        tags=["趋势型", "中频", "技术分析"],
        default_symbols=["000001", "000002", "600000", "600519"],
        parameters={
            "short_window": 10,
            "long_window": 30,
            "stop_loss_pct": 0.05
        }
    ),

    "mean_reversion": StrategyInfo(
        strategy_id="mean_reversion",
        name="均值回归策略",
        description="基于RSI和布林带的均值回归策略，捕捉超跌反弹机会。适用于震荡市场。",
        strategy_class=MeanReversionStrategy,
        tags=["均值回归", "短频", "技术分析"],
        default_symbols=["300750", "002475", "600036", "601318"],  # 宁德时代 (unique from alpha_trend)
        parameters={
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_period": 20,
            "bb_std": 2,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10
        }
    ),

    "momentum": StrategyInfo(
        strategy_id="momentum",
        name="动量策略",
        description="动量策略，使用ROC和ADX识别强势趋势。适用于趋势明显的市场。",
        strategy_class=MomentumStrategy,
        tags=["动量", "中频", "趋势跟踪"],
        default_symbols=["600519", "000858", "000333", "002304"],  # 贵州茅台 (unique)
        parameters={
            "roc_period": 20,
            "roc_threshold": 3,
            "adx_period": 14,
            "adx_trend_threshold": 25,
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.20
        }
    ),

    "sector_rotation": StrategyInfo(
        strategy_id="sector_rotation",
        name="板块轮动策略",
        description="板块轮动策略，基于相对强度在不同行业板块间配置资金。",
        strategy_class=MomentumSectorRotation,
        tags=["板块轮动", "低频", "资产配置"],
        default_symbols=["512000", "512400", "512690", "159915"],  # 行业ETF示例
        parameters={
            "lookback_period": 20,
            "top_n_sectors": 3,
            "rebalance_threshold": 0.10,
            "stop_loss_pct": 0.05
        }
    ),

    "dragon_leader": StrategyInfo(
        strategy_id="dragon_leader",
        name="龙厂策略",
        description="连板龙头打板策略，在市场情绪好时买入连板数最高的龙头股。适用于短线激进交易。",
        strategy_class=DragonLeaderStrategy,
        tags=["打板", "短线", "龙头股", "高风险"],
        default_symbols=[],  # 动态从涨停股池选择
        parameters={
            "max_hold_num": 1,
            "min_board_count": 3,
            "min_non_yz_board": 2,
            "stop_loss_pct": 0.05,
            "market_risk_threshold": 15,
            "cooldown_days": 7,
            "max_backtest_days": 60  # 回测天数（需要MA60计算，建议60天以上）
        }
    ),
}


def get_strategy_info(strategy_id: str) -> Optional[StrategyInfo]:
    """Get strategy information by ID."""
    return STRATEGY_REGISTRY.get(strategy_id)


def get_all_strategies(active_only: bool = True) -> List[StrategyInfo]:
    """Get all registered strategies."""
    strategies = list(STRATEGY_REGISTRY.values())
    if active_only:
        strategies = [s for s in strategies if s.is_active]
    return strategies


def create_strategy(strategy_id: str, **kwargs) -> Optional[Any]:
    """Create a new strategy instance."""
    info = get_strategy_info(strategy_id)
    if info:
        return info.create_instance(**kwargs)
    return None


def get_strategy_list_for_api() -> List[Dict[str, Any]]:
    """
    Get strategy list formatted for API response.

    Returns:
        List of strategy dictionaries for frontend consumption
    """
    strategies = []

    for info in get_all_strategies():
        strategies.append({
            "id": info.strategy_id,
            "name": info.name,
            "description": info.description,
            "tags": info.tags,
            "default_symbols": info.default_symbols,
            "parameters": info.parameters,
            "is_active": info.is_active
        })

    return strategies


# CSI 300 Stock Pool (sample - would be populated from database)
CSI300_SYMBOLS = [
    # 金融
    "000001",  # 平安银行
    "600000",  # 浦发银行
    "600036",  # 招商银行
    "601318",  # 中国平安
    "601398",  # 工商银行
    "601939",  # 建设银行
    "600030",  # 中信证券

    # 科技
    "300750",  # 宁德时代
    "002475",  # 立讯精密
    "000063",  # 中兴通讯
    "002415",  # 海康威视
    "600584",  # 长电科技

    # 消费
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "000333",  # 美的集团
    "002304",  # 洋河股份
    "600887",  # 伊利股份

    # 医药
    "000661",  # 长春高新
    "000538",  # 云南白药
    "600276",  # 恒瑞医药
    "300015",  # 爱尔眼科

    # 新能源
    "300124",  # 汇川技术
    "002129",  # 中环股份
    "688981",  # 中芯国际

    # 地产
    "000002",  # 万科A
    "001979",  # 招商蛇口

    # 能源
    "601857",  # 中国石油
    "600028",  # 中国石化
]


def get_stock_pool_for_strategy(strategy_id: str) -> List[str]:
    """Get appropriate stock pool for a strategy."""
    info = get_strategy_info(strategy_id)
    if info and info.default_symbols:
        return info.default_symbols
    return CSI300_SYMBOLS[:20]  # Default to first 20 CSI 300 stocks
