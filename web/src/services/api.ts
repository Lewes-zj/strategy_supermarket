import axios from 'axios';
export * from './types';

// const API_BASE = 'http://localhost:8000/api';
const API_BASE = '/api';

export const api = {
    getStrategies: async (params?: import('./types').StrategyListParams): Promise<import('./types').StrategyListItem[]> => {
        const res = await axios.get(`${API_BASE}/strategies`, { params });
        return res.data;
    },

    getMetrics: async (id: string, year?: number): Promise<import('./types').Metrics> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/metrics`, {
            params: year ? { year } : undefined
        });
        return res.data;
    },

    getEquityCurve: async (id: string, year?: number): Promise<import('./types').EquityPoint[]> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/equity_curve`, {
            params: year ? { year } : undefined
        });
        return res.data;
    },

    getTransactions: async (id: string, isSubscribed: boolean, year?: number): Promise<import('./types').Transaction[]> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/transactions`, {
            params: { is_subscribed: isSubscribed, ...(year ? { year } : {}) }
        });
        return res.data;
    },

    getHoldings: async (id: string, isSubscribed: boolean): Promise<import('./types').HoldingsResponse> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/holdings`, {
            params: { is_subscribed: isSubscribed }
        });
        return res.data;
    },

    getSignals: async (id: string): Promise<import('./types').SignalsResponse> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/signals`);
        return res.data;
    },

    getMarketSymbols: async (sector?: string): Promise<import('./types').MarketSymbol[]> => {
        const res = await axios.get(`${API_BASE}/market/symbols`, {
            params: sector ? { sector } : undefined
        });
        return res.data;
    },

    getMarketSectors: async (): Promise<import('./types').MarketSector[]> => {
        const res = await axios.get(`${API_BASE}/market/sectors`);
        return res.data;
    },

    subscribe: async (payload: { plan: string; strategy_id?: string }): Promise<any> => {
        const res = await axios.post(`${API_BASE}/user/subscribe`, payload);
        return res.data;
    },

    getSubscriptionStatus: async (): Promise<any> => {
        const res = await axios.get(`${API_BASE}/user/subscription-status`);
        return res.data;
    },

    getStrategyInfo: async (id: string): Promise<import('./types').StrategyInfo> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/info`);
        return res.data;
    },

    getYearlyData: async (id: string): Promise<import('./types').YearlyData[]> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/yearly-data`);
        return res.data;
    },

    getDrawdown: async (id: string, year?: number): Promise<import('./types').DrawdownPoint[]> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/drawdown`, {
            params: year ? { year } : undefined
        });
        return res.data;
    },

    getMonthlyReturns: async (id: string, year?: number): Promise<import('./types').MonthlyReturn[]> => {
        const res = await axios.get(`${API_BASE}/strategies/${id}/monthly-returns`, {
            params: year ? { year } : undefined
        });
        return res.data;
    }
};
