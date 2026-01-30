import React, { useEffect, useState } from 'react';
import { useOutletContext, useParams } from 'react-router-dom';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { api } from '../../services/api';
import type { EquityPoint, Metrics, DrawdownPoint, MonthlyReturn } from '../../services/types';

const DataMetrics: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const { selectedYear } = useOutletContext<{ selectedYear: number | null }>();

    // State
    const [metrics, setMetrics] = useState<Metrics | null>(null);
    const [equityData, setEquityData] = useState<EquityPoint[]>([]);
    const [drawdownData, setDrawdownData] = useState<DrawdownPoint[]>([]);
    const [monthlyReturns, setMonthlyReturns] = useState<MonthlyReturn[]>([]);

    useEffect(() => {
        const fetchData = async () => {
            if (id) {
                try {
                    // Pass selectedYear to all API calls
                    const [m, e, d, monthly] = await Promise.all([
                        api.getMetrics(id, selectedYear || undefined),
                        api.getEquityCurve(id, selectedYear || undefined),
                        api.getDrawdown(id, selectedYear || undefined),
                        api.getMonthlyReturns(id, selectedYear || undefined)
                    ]);
                    setMetrics(m);
                    setEquityData(e);
                    setDrawdownData(d);
                    setMonthlyReturns(monthly);
                } catch (err) {
                    console.error(err);
                }
            }
        };
        fetchData();
    }, [id, selectedYear]);

    if (!metrics) return <div>Loading Metrics...</div>;

    return (
        <div>
            {/* Snapshot Grid */}
            <div style={{ padding: '24px', backgroundColor: 'var(--bg-card)', borderRadius: '8px', marginBottom: '24px' }}>
                <h3 style={{ marginBottom: '24px', color: 'var(--text-secondary)', fontSize: '14px', textTransform: 'uppercase' }}>Performance Snapshot ({selectedYear})</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px', rowGap: '32px' }}>
                    {/* Row 1 */}
                    <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>夏普比率 (Sharpe)</div>
                        <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{metrics.sharpe.toFixed(2)}</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>卡玛比率 (Calmar)</div>
                        <div style={{ fontSize: '24px', fontWeight: 'bold' }}>2.30</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>盈亏比 (P/L Ratio)</div>
                        <div style={{ fontSize: '24px', fontWeight: 'bold' }}>2.18</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>平均持仓</div>
                        <div style={{ fontSize: '24px', fontWeight: 'bold' }}>17.8 Days</div>
                    </div>

                    {/* Row 2 */}
                    <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>索提诺比率</div>
                        <div style={{ fontSize: '16px' }}>3.12</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Alpha (α)</div>
                        <div style={{ fontSize: '16px' }}>0.13</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Beta (β)</div>
                        <div style={{ fontSize: '16px' }}>0.80</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>波动率</div>
                        <div style={{ fontSize: '16px' }}>15.4%</div>
                    </div>
                </div>
            </div>

            {/* Charts Area */}
            <div style={{ marginBottom: '24px' }}>
                <div style={{ backgroundColor: 'var(--bg-card)', borderRadius: '8px', padding: '24px', marginBottom: '16px' }}>
                    <h3 style={{ marginBottom: '20px', color: 'var(--text-secondary)', fontSize: '14px' }}>CUMULATIVE RETURN</h3>
                    <div style={{ height: '300px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={equityData}>
                                <defs>
                                    <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="var(--color-up-red)" stopOpacity={0.2} />
                                        <stop offset="95%" stopColor="var(--color-up-red)" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="date" hide />
                                <YAxis orientation="right" tick={{ fill: '#555' }} domain={['auto', 'auto']} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'var(--bg-main)', borderColor: 'var(--border-light)' }}
                                    itemStyle={{ color: 'white' }}
                                />
                                <Area type="monotone" dataKey="value" stroke="var(--color-up-red)" strokeWidth={2} fillOpacity={1} fill="url(#colorEquity)" />
                                <Area type="monotone" dataKey="benchmark" stroke="var(--text-secondary)" strokeWidth={1} strokeDasharray="5 5" fill="none" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div style={{ backgroundColor: 'var(--bg-card)', borderRadius: '8px', padding: '24px' }}>
                    <h3 style={{ marginBottom: '20px', color: 'var(--text-secondary)', fontSize: '14px' }}>UNDERWATER DRAWDOWN</h3>
                    <div style={{ height: '150px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={drawdownData}>
                                <defs>
                                    <linearGradient id="colorDD" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="var(--color-down-green)" stopOpacity={0} />
                                        <stop offset="95%" stopColor="var(--color-down-green)" stopOpacity={0.3} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="date" hide />
                                <YAxis orientation="right" tick={{ fill: '#555' }} domain={['auto', 0]} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'var(--bg-main)', borderColor: 'var(--border-light)' }}
                                    formatter={(value: number) => [`${value.toFixed(2)}%`, '回撤']}
                                />
                                <Area type="monotone" dataKey="drawdown" stroke="var(--color-down-green)" strokeWidth={1} fillOpacity={1} fill="url(#colorDD)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Heatmap */}
            <div style={{ padding: '24px', backgroundColor: 'var(--bg-card)', borderRadius: '8px' }}>
                <h3 style={{ marginBottom: '16px', color: 'var(--text-secondary)', fontSize: '14px' }}>MONTHLY RETURNS {selectedYear ? `(${selectedYear})` : ''}</h3>
                <div style={{ display: 'flex', gap: '4px', height: '12px' }}>
                    {Array.from({ length: 12 }).map((_, i) => {
                        const monthData = monthlyReturns.find(m => m.month === i + 1);
                        const val = monthData ? monthData.return / 100 : 0;
                        const color = val > 0 ? 'var(--color-up-red)' : val < 0 ? 'var(--color-down-green)' : 'var(--text-secondary)';
                        const opacity = monthData ? 0.3 + Math.min(Math.abs(val) * 3, 0.7) : 0.1;
                        return (
                            <div
                                key={i}
                                title={`${i + 1}月: ${monthData ? val.toFixed(1) : 'N/A'}%`}
                                style={{ flex: 1, backgroundColor: color, opacity: opacity, borderRadius: '2px' }}
                            />
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

export default DataMetrics;
