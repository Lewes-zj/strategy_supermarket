import React, { useEffect, useState } from 'react';
import { useOutletContext, useParams } from 'react-router-dom';
import { Unlock } from 'lucide-react';
import { api } from '../../services/api';
import type { Transaction } from '../../services/types';

const TransactionHistory: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const { isSubscribed, requestSubscribe, selectedYear } = useOutletContext<{
        isSubscribed: boolean;
        requestSubscribe: () => void;
        selectedYear: number | null;
    }>();
    const [transactions, setTransactions] = useState<Transaction[]>([]);

    useEffect(() => {
        const fetchData = async () => {
            if (id) {
                try {
                    const data = await api.getTransactions(id, isSubscribed, selectedYear || undefined);
                    setTransactions(data);
                } catch (e) {
                    console.error(e);
                }
            }
        };
        fetchData();
    }, [id, isSubscribed, selectedYear]);

    return (
        <div>
            {/* Toolbar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px', color: 'var(--text-secondary)', fontSize: '12px' }}>
                <div>Last Updated: Real-time</div>
                <div>Config ⚙️</div>
            </div>

            {/* Signal Alert Banner (Conditional) */}
            <div style={{
                backgroundColor: 'rgba(245, 34, 45, 0.1)', border: '1px solid var(--color-up-red)', color: 'var(--color-up-red)',
                padding: '12px', borderRadius: '4px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: 'pointer'
            }} onClick={requestSubscribe}>
                <span style={{ fontWeight: 'bold' }}>🔔 刚刚触发买入信号 (10:32)。标的：消费电子。 [点击解锁]</span>
            </div>

            {/* Table */}
            <div style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-light)' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                    <thead style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-secondary)' }}>
                        <tr>
                            <th style={{ padding: '12px', textAlign: 'left' }}>日期</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>时间</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>标的</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>方向</th>
                            <th style={{ padding: '12px', textAlign: 'right' }}>成交价</th>
                            <th style={{ padding: '12px', textAlign: 'right' }}>盈亏 (P&L)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {transactions.map((tx, i) => (
                            <tr key={i} style={{
                                borderTop: '1px solid var(--border-light)',
                                backgroundColor: tx.is_encrypted ? 'rgba(24, 144, 255, 0.02)' : 'transparent',
                                transition: 'background-color 0.2s'
                            }}
                                className={tx.is_encrypted ? "encrypted-row" : ""}
                            >
                                <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>{tx.date}</td>
                                <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>{tx.time}</td>

                                {/* Symbol Column */}
                                <td style={{ padding: '12px' }}>
                                    {tx.is_encrypted ? (
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <span style={{ background: 'var(--color-gold)', color: 'black', fontSize: '10px', padding: '2px 4px', borderRadius: '2px', fontWeight: 'bold' }}>持仓中</span>
                                            <span style={{ color: 'var(--text-primary)' }}>{tx.symbol}</span>
                                            {!isSubscribed && (
                                                <button onClick={requestSubscribe} style={{
                                                    padding: '2px 8px', fontSize: '10px', background: 'var(--color-brand-blue)',
                                                    color: 'white', border: 'none', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '4px'
                                                }}>
                                                    <Unlock size={10} /> 解锁
                                                </button>
                                            )}
                                        </div>
                                    ) : (
                                        <span style={{ color: 'white' }}>{tx.symbol}</span>
                                    )}
                                </td>

                                {/* Side */}
                                <td style={{ padding: '12px', color: tx.side === 'buy' ? 'var(--color-up-red)' : 'var(--color-down-green)', textTransform: 'uppercase' }}>
                                    {tx.side}
                                </td>

                                {/* Price */}
                                <td style={{ padding: '12px', textAlign: 'right', fontFamily: 'monospace' }}>
                                    {tx.price ? tx.price.toFixed(2) : '**.**'}
                                </td>

                                {/* P&L */}
                                <td style={{ padding: '12px', textAlign: 'right' }}>
                                    {tx.is_encrypted ? (
                                        <div onClick={requestSubscribe} style={{ cursor: 'pointer' }}>
                                            <div style={{ color: 'var(--color-up-red)', fontWeight: 'bold' }}>+{(tx.pnl_percent! * 100).toFixed(2)}%</div>
                                            <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>当前浮盈</div>
                                        </div>
                                    ) : (
                                        <span style={{ color: tx.pnl > 0 ? 'var(--color-up-red)' : 'var(--color-down-green)' }}>
                                            {tx.pnl > 0 ? '+' : ''}{(tx.pnl * 100).toFixed(2)}%
                                        </span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default TransactionHistory;
