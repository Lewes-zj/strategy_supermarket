export interface StrategyListItem {
    id: string;
    name: string;
    description?: string;
    tags: string[];
    cagr: number;
    sharpe?: number;
    win_rate: number;
    max_drawdown: number;
    sparkline: number[];
    latest_signal?: {
        has_recent: boolean;
        time: string | null;
    };
    // Additional fields from backend
    is_active?: boolean;
    total_trades?: number;
    avg_hold_days?: number;
    win_count?: number;
    recent_captures?: Array<{ symbol: string; name: string }>;
}

export interface StrategyListParams {
    search?: string;
    sort?: 'cagr' | 'sharpe' | 'max_drawdown' | 'win_rate' | 'latest_signal';
    order?: 'asc' | 'desc';
}

export interface Metrics {
    cagr: number;
    sharpe: number;
    max_drawdown: number;
    win_rate: number;
    total_return: number;
    // Additional metrics from backend
    calmar?: number;
    pl_ratio?: number;
    avg_hold_days?: number;
    sortino?: number;
    alpha?: number;
    beta?: number;
    volatility?: number;
    win_count?: number;
    loss_count?: number;
    ytd_return?: number;
    mtd_return?: number;
    consecutive_wins?: number;
    drawdown_period?: string;
}

export interface EquityPoint {
    date: string;
    value: number;
    benchmark: number;
}

export interface Transaction {
    date: string;
    time: string;
    symbol: string;
    side: string;
    price: number | null;
    pnl: number;
    pnl_percent?: number;
    is_encrypted: boolean;
}

export interface Holding {
    symbol: string;
    name: string;
    sector: string;
    direction: string;
    days_held: number;
    weight: string;
    pnl_pct: number;
    current_price: number | null;
}

export interface HoldingsResponse {
    holdings: Holding[];
    total_pnl_pct: number;
}

export interface Signal {
    id: number;
    symbol: string;
    signal_type: string;
    price: number;
    quantity: number;
    reason: string;
    created_at: string;
    is_active: boolean;
}

export interface SignalsResponse {
    strategy_id: string;
    signals: Signal[];
    has_recent: boolean;
}

export interface MarketSymbol {
    symbol: string;
    name: string;
    sector: string;
}

export interface MarketSector {
    id: string;
    name: string;
    count: number;
}

export interface StrategyInfo {
    id: string;
    name: string;
    description: string;
    tags: string[];
    is_active: boolean;
    total_metrics: Metrics;
}

export interface YearlyData {
    year: number;
    ret: number;
    is_running: boolean;
}

export interface DrawdownPoint {
    date: string;
    drawdown: number;
}

export interface MonthlyReturn {
    year: number;
    month: number;
    return: number;
}
