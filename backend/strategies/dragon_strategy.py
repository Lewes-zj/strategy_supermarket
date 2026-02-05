"""
龙厂策略 (Dragon Leader Strategy)

基于连板龙头股的打板策略。
在市场情绪好的时候，买入连板数最高的龙头股。

核心逻辑：
- 选股：从涨停股池获取连板数最高的股票
- 买入条件：
  1. 大盘趋势向上（上证指数线性拟合斜率 > 0）
  2. 连板数 >= 3
  3. 非一字板连板数 >= 2
  4. MA60 上升趋势
  5. 11点前
  6. 风控通过（跌停数未异常增加）
- 卖出条件：
  1. 尾盘未涨停（收盘价 < 涨停价）
  2. 止损5%
- 仓位管理：可用资金 / 需要买入的股票数
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


class DragonLeaderStrategy(Strategy):
    """
    龙厂策略 - 连板龙头打板策略

    Parameters:
        initial_capital: 初始资金 (默认100万)
        max_hold_num: 最多持有股票数 (默认1)
        min_board_count: 最低连板数要求 (默认3)
        min_non_yz_board: 非一字板最低连板数 (默认2)
        stop_loss_pct: 止损比例 (默认0.05, 5%)
        market_risk_threshold: 跌停数风控阈值 (默认15)
        cooldown_days: 同一股票买卖冷却期 (默认7天)
        buy_before_hour: 只在此时间前买入 (默认11点)
        ma_period: MA趋势判断周期 (默认60)
        market_trend_days: 大盘趋势拟合天数 (默认3)
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        max_hold_num: int = 1,
        min_board_count: int = 3,
        min_non_yz_board: int = 2,
        stop_loss_pct: float = 0.05,
        market_risk_threshold: int = 15,
        cooldown_days: int = 7,
        buy_before_hour: int = 11,
        ma_period: int = 60,
        market_trend_days: int = 3,
        zt_pool_service=None,
        max_backtest_days: int = None  # Used by backtest service, not strategy
    ):
        # 参数
        self.initial_capital = Decimal(str(initial_capital))
        self.max_hold_num = max_hold_num
        self.min_board_count = min_board_count
        self.min_non_yz_board = min_non_yz_board
        self.stop_loss_pct = Decimal(str(stop_loss_pct))
        self.market_risk_threshold = market_risk_threshold
        self.cooldown_days = cooldown_days
        self.buy_before_hour = buy_before_hour
        self.ma_period = ma_period
        self.market_trend_days = market_trend_days

        # 涨停股池服务（延迟加载）
        self._zt_pool_service = zt_pool_service

        # 资金和持仓跟踪
        self.cash = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # {symbol: {quantity, entry_price, entry_date}}

        # 冷却期跟踪
        self.recent_traded: Dict[str, int] = {}

        # 今日候选股
        self.today_candidates: List[Dict] = []

        # 当前交易日
        self.current_date: Optional[date] = None

        # 涨跌停价格缓存 {(ts_code, date): {up_limit, down_limit}}
        self._limit_price_cache: Dict = {}

    @property
    def zt_pool_service(self):
        """延迟加载涨停股池服务"""
        if self._zt_pool_service is None:
            from services.zt_pool_service import get_zt_pool_service
            self._zt_pool_service = get_zt_pool_service()
        return self._zt_pool_service

    def on_bar(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """处理K线数据，生成交易订单"""
        orders = []

        if data.empty:
            return orders

        current_date = timestamp.date()

        # 新的一天
        if self.current_date != current_date:
            self._on_new_day(current_date, data)
            self.current_date = current_date

        # 处理卖出
        sell_orders = self._check_sell_conditions(timestamp, data)
        orders.extend(sell_orders)

        # 处理买入
        buy_orders = self._check_buy_conditions(timestamp, data)
        orders.extend(buy_orders)

        return orders

    def on_fill(self, fill: Fill) -> None:
        """处理成交通知"""
        symbol = fill.order.symbol

        if fill.order.side == OrderSide.BUY:
            # 更新持仓
            cost = fill.fill_price * fill.fill_quantity + fill.commission
            self.cash -= cost

            if symbol in self.positions:
                # 加仓
                pos = self.positions[symbol]
                total_qty = pos['quantity'] + fill.fill_quantity
                pos['entry_price'] = (
                    (pos['quantity'] * pos['entry_price'] + fill.fill_quantity * fill.fill_price)
                    / total_qty
                )
                pos['quantity'] = total_qty
            else:
                self.positions[symbol] = {
                    'quantity': fill.fill_quantity,
                    'entry_price': fill.fill_price,
                    'entry_date': fill.timestamp.date()
                }

            self.recent_traded[symbol] = 0

        elif fill.order.side == OrderSide.SELL:
            # 更新资金
            proceeds = fill.fill_price * fill.fill_quantity - fill.commission
            self.cash += proceeds

            # 更新持仓
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos['quantity'] -= fill.fill_quantity
                if pos['quantity'] <= 0:
                    del self.positions[symbol]

            self.recent_traded[symbol] = 0

    def _on_new_day(self, current_date: date, data: pd.DataFrame) -> None:
        """新交易日处理"""
        # 更新冷却期
        to_remove = []
        for symbol, days in self.recent_traded.items():
            self.recent_traded[symbol] = days + 1
            if days + 1 > self.cooldown_days:
                to_remove.append(symbol)
        for symbol in to_remove:
            del self.recent_traded[symbol]

        # 盘前选股
        self.today_candidates = self._pre_market_selection(current_date, data)

    def _pre_market_selection(self, current_date: date, data: pd.DataFrame) -> List[Dict]:
        """
        盘前选股逻辑

        使用 T-1 日的涨停股池数据，筛选最高板股票。
        """
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
            ts_code = stock.get('ts_code', self._to_ts_code(symbol))

            # 排除科创板 (688)
            if symbol.startswith('688'):
                continue

            # 排除冷却期内的股票
            if symbol in self.recent_traded:
                continue

            # 排除一字板（无法买入）
            if stock.get('is_yizi', False):
                continue

            # 检查非一字板连板数
            # 如果连板数 >= min_non_yz_board 且不是一字板，则符合条件
            if stock['board_count'] >= self.min_non_yz_board:
                # MA60 趋势过滤
                if self._check_ma_trend(symbol, data):
                    filtered.append(stock)

        return filtered

    def _check_ma_trend(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        检查 MA60 趋势

        条件：MA60 今天 > MA60 昨天（MA60 上升）
        """
        symbol_data = data[data['symbol'] == symbol] if 'symbol' in data.columns else data

        if symbol_data.empty or len(symbol_data) < self.ma_period + 1:
            return True  # 数据不足，默认通过

        closes = symbol_data['close'].values

        if len(closes) < self.ma_period + 1:
            return True

        # MA60 今天
        ma_today = np.mean(closes[-self.ma_period:])
        # MA60 昨天
        ma_yesterday = np.mean(closes[-(self.ma_period + 1):-1])

        return ma_today > ma_yesterday

    def _check_market_trend(self, data: pd.DataFrame) -> bool:
        """
        检查大盘趋势

        使用上证指数的最近N天收盘价进行线性拟合。
        斜率 > 0 认为趋势向上。
        """
        # 尝试获取上证指数数据
        market_symbol = None
        for sym in ['000001.SH', '000001']:
            if sym in data['symbol'].values:
                market_symbol = sym
                break

        if market_symbol is None:
            # 使用第一个标的作为市场代理
            symbols = data['symbol'].unique()
            if len(symbols) == 0:
                return False
            market_symbol = symbols[0]

        market_data = data[data['symbol'] == market_symbol]

        if len(market_data) < self.market_trend_days:
            return True  # 数据不足，默认允许

        closes = market_data['close'].tail(self.market_trend_days).values

        # 线性拟合
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        return slope > 0

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

    def _get_limit_prices(self, ts_code: str, trade_date: date) -> Dict:
        """获取涨跌停价格"""
        cache_key = (ts_code, trade_date)
        if cache_key in self._limit_price_cache:
            return self._limit_price_cache[cache_key]

        from database.connection import get_session
        from sqlalchemy import text

        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT up_limit, down_limit
                    FROM tushare_stock_limit
                    WHERE ts_code = :ts_code AND trade_date = :trade_date
                """),
                {'ts_code': ts_code, 'trade_date': trade_date}
            )
            row = result.fetchone()

        if row:
            limits = {'up_limit': row[0], 'down_limit': row[1]}
        else:
            limits = {'up_limit': None, 'down_limit': None}

        self._limit_price_cache[cache_key] = limits
        return limits

    def _check_sell_conditions(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """检查卖出条件"""
        orders = []

        for symbol, pos_info in list(self.positions.items()):
            symbol_data = data[data['symbol'] == symbol]
            if symbol_data.empty:
                continue

            current_price = Decimal(str(symbol_data['close'].iloc[-1]))
            entry_price = pos_info['entry_price']
            entry_date = pos_info['entry_date']
            quantity = pos_info['quantity']

            should_sell = False
            reason = ""

            # 条件1: 止损 (亏损超过5%)
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct < -self.stop_loss_pct:
                should_sell = True
                reason = "stop_loss"

            # 条件2: 尾盘未涨停
            # 检查收盘价是否 < 涨停价
            if not should_sell:
                days_held = (timestamp.date() - entry_date).days
                if days_held >= 1:  # 持仓超过1天才检查
                    ts_code = self._to_ts_code(symbol)
                    limits = self._get_limit_prices(ts_code, timestamp.date())
                    up_limit = limits.get('up_limit')

                    if up_limit is not None:
                        # 收盘价 < 涨停价 * 0.998，认为未涨停
                        if float(current_price) < up_limit * 0.998:
                            should_sell = True
                            reason = "not_limit_up_at_close"

            if should_sell:
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                ))

        return orders

    def _check_buy_conditions(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        """检查买入条件"""
        orders = []

        # 检查持仓数
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

        # 计算可买入的股票数量
        need_count = self.max_hold_num - len(self.positions)
        if need_count <= 0:
            return orders

        # 计算每只股票的买入金额
        buy_cash_per_stock = self.cash / Decimal(str(need_count))

        # 尝试买入候选股（按连板数排序，优先买最高板）
        for candidate in self.today_candidates:
            if len(self.positions) >= self.max_hold_num:
                break

            symbol = candidate['symbol']
            ts_code = candidate.get('ts_code', self._to_ts_code(symbol))

            # 已持有则跳过
            if symbol in self.positions:
                continue

            # 检查该股票今日是否有数据
            symbol_data = data[data['symbol'] == symbol]
            if symbol_data.empty:
                # 尝试用 ts_code 查找
                symbol_data = data[data['symbol'] == ts_code]
                if symbol_data.empty:
                    continue

            current_price = Decimal(str(symbol_data['close'].iloc[-1]))

            # 检查是否已涨停（不追涨停）
            limits = self._get_limit_prices(ts_code, timestamp.date())
            up_limit = limits.get('up_limit')

            if up_limit is not None:
                if float(current_price) >= up_limit * 0.998:
                    continue  # 已涨停，跳过

            # 计算买入数量（按手为单位，1手=100股）
            if current_price <= 0:
                continue

            # 确保有足够资金买入至少1手
            min_buy_amount = current_price * Decimal("100")
            if buy_cash_per_stock < min_buy_amount:
                continue

            # 计算可买入的手数
            lots = int(buy_cash_per_stock / (current_price * Decimal("100")))
            if lots <= 0:
                continue

            quantity = Decimal(str(lots * 100))

            orders.append(Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                timestamp=timestamp
            ))

            # 一次只买一只
            break

        return orders

    def _to_ts_code(self, symbol: str) -> str:
        """转换为 tushare 代码格式"""
        if '.' in symbol:
            return symbol.upper()
        if symbol.startswith(('6', '9')):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    def prepare_backtest_data(self, start_date: date, end_date: date) -> Dict:
        """准备回测所需数据"""
        return self.zt_pool_service.prepare_for_backtest(start_date, end_date)

    def reset(self):
        """重置策略状态（用于多次回测）"""
        self.cash = self.initial_capital
        self.positions = {}
        self.recent_traded = {}
        self.today_candidates = []
        self.current_date = None
        self._limit_price_cache = {}


def create_dragon_strategy(**kwargs) -> DragonLeaderStrategy:
    """创建龙厂策略实例"""
    return DragonLeaderStrategy(**kwargs)
