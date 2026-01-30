import React, { useEffect, useState } from 'react';
import { useOutletContext, useParams } from 'react-router-dom';
import { Lock, Shield } from 'lucide-react';
import { api } from '../../services/api';
import type { Holding, HoldingsResponse } from '../../services/types';

const DailyHoldings: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const { isSubscribed, requestSubscribe } = useOutletContext<{ isSubscribed: boolean, requestSubscribe: () => void }>();
    const [holdings, setHoldings] = useState<Holding[]>([]);
    const [totalPnl, setTotalPnl] = useState(0);

    useEffect(() => {
        const fetchData = async () => {
            if (id) {
                try {
                    const data = await api.getHoldings(id, isSubscribed);
                    setHoldings(data.holdings);
                    setTotalPnl(data.total_pnl_pct);
                } catch (e) {
                    console.error(e);
                }
            }
        };
        fetchData();
    }, [id, isSubscribed]);

    return (
        <div style={{ position: 'relative', minHeight: '400px' }}>
            {/* Layer 1: The List (Background) */}
            <div style={{
                filter: isSubscribed ? 'none' : 'blur(6px)',
                pointerEvents: isSubscribed ? 'auto' : 'none',
                transition: 'filter 0.5s ease',
                userSelect: isSubscribed ? 'auto' : 'none'
            }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                    <thead style={{ borderBottom: '1px solid var(--border-light)', color: 'var(--text-secondary)' }}>
                        <tr>
                            <th style={{ padding: '12px', textAlign: 'left' }}>代码/名称</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>行业板块</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>方向</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>持仓天数</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>仓位占比</th>
                            <th style={{ padding: '12px', textAlign: 'right' }}>累计浮盈</th>
                        </tr>
                    </thead>
                    <tbody>
                        {holdings.map((h, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <td style={{ padding: '16px 12px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                        <Lock size={14} className="text-gold" />
                                        <span style={{
                                            backgroundColor: isSubscribed ? 'transparent' : 'rgba(255,255,255,0.1)',
                                            color: isSubscribed ? 'white' : 'transparent',
                                            borderRadius: '4px', padding: '2px 4px'
                                        }}>
                                            {h.symbol} {h.name}
                                        </span>
                                    </div>
                                </td>
                                <td style={{ padding: '12px' }}>{h.sector}</td>
                                <td style={{ padding: '12px' }}>
                                    <span style={{
                                        color: h.direction === 'Long' ? 'var(--color-up-red)' : 'var(--color-down-green)',
                                        border: '1px solid currentColor', padding: '2px 6px', borderRadius: '4px', fontSize: '10px'
                                    }}>{h.direction}</span>
                                </td>
                                <td style={{ padding: '12px' }}>{h.days_held}天</td>
                                <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>{h.weight}</td>
                                <td style={{ padding: '12px', textAlign: 'right', fontWeight: 'bold', color: h.pnl_pct > 0 ? 'var(--color-up-red)' : 'var(--color-down-green)' }}>
                                    {h.pnl_pct > 0 ? '+' : ''}{(h.pnl_pct * 100).toFixed(1)}%
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Layer 2: The Overlay (Foreground) */}
            {!isSubscribed && (
                <div style={{
                    position: 'absolute', top: '100px', left: 0, right: 0, bottom: 0,
                    background: 'linear-gradient(to bottom, rgba(20, 24, 32, 0) 0%, rgba(20, 24, 32, 0.95) 40%, rgba(20, 24, 32, 1) 100%)',
                    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                    zIndex: 10
                }}>
                    <div style={{ fontSize: '64px', fontWeight: 'bold', color: '#D14646', fontFamily: 'DIN Condensed', textShadow: '0 0 20px rgba(209, 70, 70, 0.3)' }}>
                        +{(totalPnl * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '24px' }}>
                        当前持仓 {holdings.length} 只，策略实盘运行中
                    </div>
                    <button
                        onClick={requestSubscribe}
                        style={{
                            background: 'var(--color-brand-blue)', color: 'white', border: 'none',
                            padding: '16px 48px', fontSize: '18px', fontWeight: 'bold', borderRadius: '8px',
                            marginBottom: '16px', boxShadow: '0 4px 12px rgba(24, 144, 255, 0.4)'
                        }}
                    >
                        <Lock size={18} style={{ marginRight: '8px', verticalAlign: 'text-bottom' }} />
                        立即解锁全部代码
                    </button>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '12px' }}>
                        <Shield size={14} /> 7天无理由退款
                    </div>
                </div>
            )}
        </div>
    );
};

export default DailyHoldings;
