# backend/engine/metrics.py
"""
绩效指标计算模块
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from .models import PerformanceMetrics, Fill, OrderSide


def calculate_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    ann_factor: int = 252
) -> float:
    """计算夏普比率"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / ann_factor
    return float(excess_returns.mean() / returns.std() * np.sqrt(ann_factor))


def calculate_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    ann_factor: int = 252
) -> float:
    """计算索提诺比率"""
    if len(returns) == 0:
        return 0.0

    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    ann_return = (1 + returns.mean()) ** ann_factor - 1
    downside_std = downside.std() * np.sqrt(ann_factor)

    return float((ann_return - risk_free_rate) / downside_std)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤"""
    if len(returns) == 0:
        return 0.0

    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max

    return float(drawdowns.min())


def calculate_calmar(returns: pd.Series, ann_factor: int = 252) -> float:
    """计算卡玛比率"""
    if len(returns) == 0:
        return 0.0

    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / ann_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    max_dd = calculate_max_drawdown(returns)

    return float(cagr / abs(max_dd)) if max_dd != 0 else 0.0


def calculate_alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    ann_factor: int = 252
) -> Tuple[float, float]:
    """计算Alpha和Beta"""
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0, 1.0

    # 对齐数据
    aligned = pd.DataFrame({
        "strategy": returns,
        "benchmark": benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        return 0.0, 1.0

    # 计算Beta
    cov = aligned["strategy"].cov(aligned["benchmark"])
    var = aligned["benchmark"].var()
    beta = cov / var if var > 0 else 1.0

    # 计算策略和基准的年化收益
    strat_total = (1 + aligned["strategy"]).prod() - 1
    bench_total = (1 + aligned["benchmark"]).prod() - 1
    n_years = len(aligned) / ann_factor

    strat_cagr = (1 + strat_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    bench_cagr = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Alpha = 策略收益 - (无风险 + Beta * (基准收益 - 无风险))
    alpha = strat_cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))

    return float(alpha), float(beta)


def calculate_win_rate(returns: pd.Series) -> Tuple[float, int, int]:
    """计算胜率"""
    if len(returns) == 0:
        return 0.0, 0, 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_count = len(wins)
    loss_count = len(losses)
    total = win_count + loss_count

    win_rate = win_count / total if total > 0 else 0.0

    return float(win_rate), win_count, loss_count


def calculate_pl_ratio(returns: pd.Series) -> float:
    """计算盈亏比"""
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0001

    return float(avg_win / avg_loss) if avg_loss > 0 else 0.0


def calculate_consecutive_wins(returns: pd.Series) -> int:
    """计算最大连续获胜天数"""
    max_streak = 0
    current_streak = 0

    for ret in returns:
        if ret > 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def calculate_period_returns(returns: pd.Series) -> Tuple[float, float]:
    """计算YTD和MTD收益"""
    if len(returns) == 0:
        return 0.0, 0.0

    current_date = returns.index[-1]

    # YTD
    try:
        ytd_start = current_date.replace(month=1, day=1)
        ytd_returns = returns[returns.index >= ytd_start]
        ytd = (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0
    except:
        ytd = 0.0

    # MTD
    try:
        mtd_start = current_date.replace(day=1)
        mtd_returns = returns[returns.index >= mtd_start]
        mtd = (1 + mtd_returns).prod() - 1 if len(mtd_returns) > 0 else 0
    except:
        mtd = 0.0

    return float(ytd), float(mtd)


def calculate_avg_hold_days(trades: List[Fill]) -> float:
    """从交易记录计算平均持仓天数"""
    if not trades:
        return 0.0

    positions = {}  # symbol -> entry_date
    hold_days = []

    for trade in trades:
        symbol = trade.order.symbol
        if trade.order.side == OrderSide.BUY:
            positions[symbol] = trade.timestamp.date()
        elif trade.order.side == OrderSide.SELL and symbol in positions:
            entry = positions[symbol]
            exit_date = trade.timestamp.date()
            days = (exit_date - entry).days
            if days > 0:
                hold_days.append(days)
            del positions[symbol]

    return sum(hold_days) / len(hold_days) if hold_days else 0.0


def calculate_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    trades: List[Fill] = None,
    risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """
    计算完整绩效指标

    Args:
        returns: 策略日收益率
        benchmark_returns: 基准日收益率
        trades: 交易记录（用于计算交易统计）
        risk_free_rate: 无风险利率

    Returns:
        PerformanceMetrics
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return PerformanceMetrics()

    ann_factor = 252

    # 基础收益指标
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / ann_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = returns.std() * np.sqrt(ann_factor)

    # 风险调整指标
    sharpe = calculate_sharpe(returns, risk_free_rate, ann_factor)
    sortino = calculate_sortino(returns, risk_free_rate, ann_factor)
    max_drawdown = calculate_max_drawdown(returns)
    calmar = calculate_calmar(returns, ann_factor)

    # Alpha/Beta
    alpha, beta = 0.0, 1.0
    benchmark_return = 0.0
    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(returns, benchmark_returns, risk_free_rate, ann_factor)
        bench_total = (1 + benchmark_returns.dropna()).prod() - 1
        bench_years = len(benchmark_returns.dropna()) / ann_factor
        benchmark_return = (1 + bench_total) ** (1 / bench_years) - 1 if bench_years > 0 else 0

    # 胜率
    win_rate, win_count, loss_count = calculate_win_rate(returns)
    pl_ratio = calculate_pl_ratio(returns)
    consecutive_wins = calculate_consecutive_wins(returns)

    # 时间段收益
    ytd_return, mtd_return = calculate_period_returns(returns)

    # 回撤区间
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd_date = drawdowns.idxmin()
    drawdown_period = max_dd_date.strftime("%Y/%m") if pd.notna(max_dd_date) else "N/A"

    # 平均持仓天数
    avg_hold_days = calculate_avg_hold_days(trades) if trades else 0.0

    return PerformanceMetrics(
        sharpe=sharpe,
        calmar=calmar,
        sortino=sortino,
        max_drawdown=max_drawdown,
        total_return=float(total_return),
        cagr=float(cagr),
        ytd_return=ytd_return,
        mtd_return=mtd_return,
        volatility=float(volatility),
        alpha=alpha,
        beta=beta,
        win_rate=win_rate,
        win_count=win_count,
        loss_count=loss_count,
        pl_ratio=pl_ratio,
        avg_hold_days=avg_hold_days,
        consecutive_wins=consecutive_wins,
        benchmark_return=float(benchmark_return),
        excess_return=float(cagr - benchmark_return),
        drawdown_period=drawdown_period
    )
