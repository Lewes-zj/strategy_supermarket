import React from 'react';
import { useNavigate } from 'react-router-dom';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';
import { Target } from 'lucide-react';
import type { StrategyListItem } from '../services/types';

interface StrategyCardProps {
    strategy: StrategyListItem;
}

const StrategyCard: React.FC<StrategyCardProps> = ({ strategy }) => {
    const navigate = useNavigate();

    // Transform sparkline array to object array for Recharts
    const chartData = strategy.sparkline.map((val, i) => ({ i, val }));

    // Format recent captures for display
    const recentCapturesText = strategy.recent_captures && strategy.recent_captures.length > 0
        ? strategy.recent_captures.map(c => `${c.name}(${c.symbol})`).join('、')
        : '暂无最近交易';

    return (
        <div
            onClick={() => navigate(`/strategy/${strategy.id}`)}
            className="strategy-card"
            style={{
                backgroundColor: 'var(--bg-card)',
                borderRadius: '8px',
                padding: '20px',
                marginBottom: '16px',
                cursor: 'pointer',
                transition: 'transform 0.2s, box-shadow 0.2s',
                border: '1px solid transparent'
            }}
            onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = '0 10px 30px -10px rgba(0,0,0,0.5)';
                e.currentTarget.style.borderColor = 'var(--border-light)';
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
                e.currentTarget.style.borderColor = 'transparent';
            }}
        >
            {/* Header Line */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div>
                        <h3 style={{ fontSize: '18px', color: 'white', marginBottom: '2px' }}>{strategy.name}</h3>
                        {strategy.description && (
                            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0 }}>
                                {strategy.description}
                            </p>
                        )}
                    </div>
                    {strategy.is_active !== false && (
                        <span style={{
                            color: 'var(--color-down-green)', fontSize: '12px',
                            display: 'flex', alignItems: 'center', gap: '4px'
                        }}>
                            ● 实盘运行中
                        </span>
                    )}
                    {strategy.is_active === false && (
                        <span style={{
                            color: 'var(--text-secondary)', fontSize: '12px',
                            display: 'flex', alignItems: 'center', gap: '4px'
                        }}>
                            ● 已停止
                        </span>
                    )}
                    {strategy.tags.map(tag => (
                        <span key={tag} style={{
                            fontSize: '12px', backgroundColor: 'rgba(255,255,255,0.1)',
                            padding: '2px 8px', borderRadius: '100px', color: 'var(--text-secondary)'
                        }}>
                            {tag}
                        </span>
                    ))}
                </div>

                {/* Signal Radar */}
                {strategy.latest_signal?.has_recent ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--color-up-red)' }}>
                        <div style={{
                            width: '8px', height: '8px', backgroundColor: 'var(--color-up-red)',
                            borderRadius: '50%', animation: 'pulse-red 1.5s infinite'
                        }}></div>
                        <span style={{ fontSize: '12px', fontWeight: 'bold' }}>
                            最新信号：{strategy.latest_signal.time || '今天 09:30'}
                        </span>
                    </div>
                ) : (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)' }}>
                        <div style={{
                            width: '8px', height: '8px', backgroundColor: 'var(--text-secondary)',
                            borderRadius: '50%'
                        }}></div>
                        <span style={{ fontSize: '12px' }}>等待信号</span>
                    </div>
                )}
            </div>

            {/* Data Body */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.5fr 1fr', gap: '24px', alignItems: 'center' }}>
                {/* Left: CAGR */}
                <div>
                    <div style={{ fontSize: '36px', fontWeight: 'bold', color: strategy.cagr >= 0 ? 'var(--color-up-red)' : 'var(--color-down-green)', fontFamily: 'DIN Condensed, sans-serif' }}>
                        {strategy.cagr >= 0 ? '+' : ''}{(strategy.cagr * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>年化收益率 (CAGR)</div>
                </div>

                {/* Middle: Sparkline */}
                <div style={{ height: '60px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                            <defs>
                                <linearGradient id={`gradient-${strategy.id}`} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#F5222D" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#F5222D" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <Area
                                type="monotone"
                                dataKey="val"
                                stroke="#F5222D"
                                strokeWidth={2}
                                fill={`url(#gradient-${strategy.id})`}
                                isAnimationActive={false}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Right: Metrics */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', rowGap: '8px', columnGap: '16px' }}>
                    <div>
                        <div className="text-red font-bold">{(strategy.win_rate * 100).toFixed(1)}%</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>胜率</div>
                    </div>
                    <div>
                        <div className="text-green font-bold">{(strategy.max_drawdown * 100).toFixed(1)}%</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>最大回撤</div>
                    </div>
                    {strategy.sharpe !== undefined && (
                        <>
                            <div>
                                <div style={{ color: 'white' }}>{strategy.sharpe.toFixed(2)}</div>
                                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>夏普比率</div>
                            </div>
                            <div>
                                <div style={{ color: 'white' }}>
                                    {strategy.avg_hold_days !== undefined && strategy.avg_hold_days > 0
                                        ? `${strategy.avg_hold_days.toFixed(1)}天`
                                        : '—'}
                                </div>
                                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>平均持仓</div>
                            </div>
                        </>
                    )}
                    {strategy.sharpe === undefined && (
                        <>
                            <div>
                                <div style={{ color: 'white' }}>
                                    {strategy.avg_hold_days !== undefined && strategy.avg_hold_days > 0
                                        ? `${strategy.avg_hold_days.toFixed(1)}天`
                                        : '—'}
                                </div>
                                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>平均持仓</div>
                            </div>
                            <div>
                                <div style={{ color: 'white' }}>
                                    {strategy.win_count !== undefined ? `${strategy.win_count}笔` : '—'}
                                </div>
                                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>盈利次数</div>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Footer */}
            <div style={{
                marginTop: '20px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.05)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                    <Target size={14} className="text-blue" />
                    <span>最近捕获：{recentCapturesText}</span>
                </div>
                <button style={{
                    background: 'transparent', border: '1px solid var(--text-secondary)',
                    color: 'var(--text-secondary)', padding: '4px 12px', fontSize: '12px',
                    borderRadius: '4px'
                }}>
                    分析实盘数据 →
                </button>
            </div>
        </div>
    );
};

export default StrategyCard;
