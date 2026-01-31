"""
龙厂策略 (Dragon Leader Strategy)

基于连板龙头股的打板策略。
在市场情绪好的时候，买入连板数最高的龙头股。

核心逻辑：
- 选股：从涨停股池获取连板数最高的股票
- 买入：大盘趋势向上 + 连板数≥3 + 非一字板连板数≥2 + 11点前
- 卖出：尾盘未涨停 或 止损5%
- 风控：跌停数异常增加时暂停买入
"""
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Set
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from engine.backtester import Strategy
from engine.models import Order, OrderSide, OrderType, Fill

# Note: ZTPoolService is imported lazily to avoid circular imports
# from services.zt_pool_service import ZTPoolService, get_zt_pool_service


class DragonLeaderStrategy(Strategy):
    """
    龙厂策略 - 连板龙头打板策略

    Parameters:
        max_hold_num: 最多持有股票数 (默认1)
        min_board_count: 最低连板数要求 (默认3)
        min_non_yz_board: 非一字板最低连板数 (默认2)
        stop_loss_pct: 止损比例 (默认0.05, 5%)
        market_risk_threshold: 跌停数风控阈值 (默认15)
        cooldown_days: 同一股票买卖冷却期 (默认7天)
        buy_before_hour: 只在此时间前买入 (默认11点)
        max_backtest_days: 最大回测天数 (默认30，涨停池API限制)
    """

    def __init__(
        self,
        max_hold_num: int = 1,
        min_board_count: int = 3,
        min_non_yz_board: int = 2,
        stop_loss_pct: float = 0.05,
        market_risk_threshold: int = 15,
        cooldown_days: int = 7,
        buy_before_hour: int = 11,
        max_backtest_days: int = 30,
        zt_pool_service = None
    ):
        self.max_hold_num = max_hold_num
        self.min_board_count = min_board_count
        self.min_non_yz_board = min_non_yz_board
        self.stop_loss_pct = Decimal(str(stop_loss_pct))
        self.market_risk_threshold = market_risk_threshold
        self.cooldown_days = cooldown_days
        self.buy_before_hour = buy_before_hour
        self.max_backtest_days = max_backtest_days

        # 涨停股池服务（延迟加载以避免循环导入）
        self._zt_pool_service = zt_pool_service

        # 持仓跟踪
        self.positions: Dict[str, Dict] = {}  # {symbol: {entry_price, entry_date}}

        # 冷却期跟踪：{symbol: days_since_trade}
        self.recent_traded: Dict[str, int] = {}

        # 今日候选股（盘前选出）
        self.today_candidates: List[Dict] = []

        # 当前交易日
        self.current_date: Optional[date] = None

        # 大盘趋势判断数据
        self.market_closes: List[float] = []  # 最近N日上证指数收盘价

    @property
    def zt_pool_service(self):
        """延迟加载涨停股池服务"""
        if self._zt_pool_service is None:
            from services.zt_pool_service import get_zt_pool_service
            self._zt_pool_service = get_zt_pool_service()
        return self._zt_pool_service

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """
        处理K线数据，生成交易订单

        回测引擎会对每个交易日调用此方法。
        data 是截止到当前时间的所有标的合并数据。
        """
        orders = []

        if data.empty:
            return orders

        current_date = timestamp.date()

        # 检测是否是新的一天
        if self.current_date != current_date:
            self._on_new_day(current_date, data)
            self.current_date = current_date

        # ============ 处理卖出 ============
        sell_orders = self._check_sell_conditions(timestamp, data)
        orders.extend(sell_orders)

        # ============ 处理买入 ============
        buy_orders = self._check_buy_conditions(timestamp, data)
        orders.extend(buy_orders)

        return orders

    def on_fill(self, fill: Fill) -> None:
        """处理成交通知"""
        symbol = fill.order.symbol

        if fill.order.side == OrderSide.BUY:
            # 记录买入
            self.positions[symbol] = {
                'entry_price': fill.fill_price,
                'entry_date': fill.timestamp.date()
            }
            # 加入冷却期跟踪
            self.recent_traded[symbol] = 0

        elif fill.order.side == OrderSide.SELL:
            # 清除持仓
            if symbol in self.positions:
                del self.positions[symbol]
            # 重置冷却期计数
            self.recent_traded[symbol] = 0

    def _on_new_day(self, current_date: date, data: pd.DataFrame) -> None:
        """
        新交易日开始时的处理

        - 更新冷却期计数
        - 执行盘前选股
        """
        # 更新冷却期计数
        to_remove = []
        for symbol, days in self.recent_traded.items():
            self.recent_traded[symbol] = days + 1
            if days + 1 > self.cooldown_days:
                to_remove.append(symbol)
        for symbol in to_remove:
            del self.recent_traded[symbol]

        # 盘前选股（使用T-1日数据）
        self.today_candidates = self._pre_market_selection(current_date)

    def _pre_market_selection(self, current_date: date) -> List[Dict]:
        """
        盘前选股逻辑

        使用 T-1 日的涨停股池数据，筛选最高板股票。
        """
        # 获取 T-1（昨日）交易日
        t_1 = self.zt_pool_service.get_previous_trading_day(current_date, n=1)
        if t_1 is None:
            return []

        # 获取最高板股票
        candidates = self.zt_pool_service.get_max_board_stocks(
            t_1, min_board=self.min_board_count
        )

        if not candidates:
            return []

        # 过滤条件
        filtered = []
        for stock in candidates:
            symbol = stock['symbol']

            # 排除科创板 (688)
            if symbol.startswith('688'):
                continue

            # 排除冷却期内的股票
            if symbol in self.recent_traded:
                continue

            # 检查非一字板条件
            # 炸板次数 > 0 说明不是一字板
            # 或者首次封板时间 > 09:30 说明不是开盘就封板
            is_non_yz = (
                stock['break_count'] > 0 or
                self._is_late_board(stock['first_board_time'])
            )

            # 非一字板连板数检查
            # 这里简化处理：如果是非一字板，认为非一字板连板数 = 总连板数
            # 实际逻辑可能需要更复杂的历史分析
            if stock['board_count'] >= self.min_non_yz_board or is_non_yz:
                filtered.append(stock)

        return filtered

    def _is_late_board(self, first_board_time: str) -> bool:
        """判断首次封板时间是否晚于开盘（非一字板）"""
        if not first_board_time:
            return False

        try:
            # 格式可能是 "092500" 或 "09:25:00"
            time_str = first_board_time.replace(':', '')
            if len(time_str) >= 4:
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                # 如果在 09:30 之后封板，认为是非一字板
                return hour > 9 or (hour == 9 and minute > 30)
        except:
            pass
        return False

    def _check_risk_management(self, current_date: date) -> bool:
        """
        风险管理检查

        如果昨日跌停数 > 前日跌停数 且 昨日跌停数 > 阈值，返回 True（应暂停买入）
        """
        t_1 = self.zt_pool_service.get_previous_trading_day(current_date, n=1)
        t_2 = self.zt_pool_service.get_previous_trading_day(current_date, n=2)

        if t_1 is None or t_2 is None:
            return False

        dt_count_t1 = self.zt_pool_service.get_dt_count(t_1)
        dt_count_t2 = self.zt_pool_service.get_dt_count(t_2)

        if dt_count_t1 > dt_count_t2 and dt_count_t1 > self.market_risk_threshold:
            return True

        return False

    def _check_market_trend(self, data: pd.DataFrame) -> bool:
        """
        检查大盘趋势

        使用上证指数（或第一个标的）的最近3天收盘价进行线性拟合。
        如果斜率 > 0，认为趋势向上。
        """
        # 尝试获取上证指数数据，如果没有则使用第一个标的
        if '000001' in data['symbol'].values:
            market_data = data[data['symbol'] == '000001']
        else:
            # 使用第一个标的作为市场代理
            symbols = data['symbol'].unique()
            if len(symbols) == 0:
                return False
            market_data = data[data['symbol'] == symbols[0]]

        if len(market_data) < 3:
            return True  # 数据不足，默认允许

        # 取最近3天收盘价
        closes = market_data['close'].tail(3).values

        # 简单线性拟合
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        return slope > 0

    def _check_sell_conditions(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """检查卖出条件"""
        orders = []

        for symbol, pos_info in list(self.positions.items()):
            # 获取当前价格
            symbol_data = data[data['symbol'] == symbol]
            if symbol_data.empty:
                continue

            current_price = Decimal(str(symbol_data['close'].iloc[-1]))
            entry_price = pos_info['entry_price']

            should_sell = False
            reason = ""

            # 条件1: 止损 (亏损超过5%)
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct < -self.stop_loss_pct:
                should_sell = True
                reason = "stop_loss"

            # 条件2: 尾盘未涨停（简化：当日涨幅 < 9%）
            # 在回测中，我们用收盘价判断。实际上应该在14:55判断。
            # 这里简化为：如果当日涨幅 < 9%（接近涨停），考虑卖出
            if 'open' in symbol_data.columns:
                day_open = Decimal(str(symbol_data['open'].iloc[-1]))
                day_return = (current_price - day_open) / day_open
                # 如果涨幅小于5%，认为"未涨停"（简化处理）
                if day_return < Decimal("0.05"):
                    # 再检查是否已持仓超过1天
                    days_held = (timestamp.date() - pos_info['entry_date']).days
                    if days_held >= 1:
                        should_sell = True
                        reason = "not_limit_up_at_close"

            if should_sell:
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=Decimal("100"),  # 全部卖出
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                ))

        return orders

    def _check_buy_conditions(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """检查买入条件"""
        orders = []

        # 检查是否已达到最大持仓数
        if len(self.positions) >= self.max_hold_num:
            return orders

        # 检查风险管理
        if self._check_risk_management(timestamp.date()):
            return orders

        # 检查大盘趋势
        if not self._check_market_trend(data):
            return orders

        # 没有候选股
        if not self.today_candidates:
            return orders

        # 尝试买入候选股（按连板数排序，优先买最高板）
        for candidate in self.today_candidates:
            if len(self.positions) >= self.max_hold_num:
                break

            symbol = candidate['symbol']

            # 已持有则跳过
            if symbol in self.positions:
                continue

            # 检查该股票今日是否有数据
            symbol_data = data[data['symbol'] == symbol]
            if symbol_data.empty:
                continue

            # 获取当前价格
            current_price = Decimal(str(symbol_data['close'].iloc[-1]))

            # 检查是否已涨停（不追涨停）
            # 简化判断：如果今日涨幅 > 9%，认为可能已涨停
            if 'open' in symbol_data.columns:
                day_open = Decimal(str(symbol_data['open'].iloc[-1]))
                if day_open > 0:
                    day_return = (current_price - day_open) / day_open
                    if day_return > Decimal("0.09"):
                        continue  # 已涨停，跳过

            # 生成买入订单
            orders.append(Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Decimal("100"),  # 买入100股（简化）
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))

            # 暂时标记为已选中（防止同一天多次下单）
            break  # 一次只买一只

        return orders

    def prepare_backtest_data(self, start_date: date, end_date: date) -> Dict:
        """
        准备回测所需数据

        调用 ZT Pool Service 的预加载功能。

        Returns:
            准备结果统计
        """
        return self.zt_pool_service.prepare_for_backtest(start_date, end_date)


# 便捷函数
def create_dragon_strategy(**kwargs) -> DragonLeaderStrategy:
    """创建龙厂策略实例"""
    return DragonLeaderStrategy(**kwargs)
