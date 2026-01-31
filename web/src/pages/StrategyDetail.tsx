import React, { useState, useEffect } from 'react';
import { NavLink, Outlet, useParams } from 'react-router-dom';
import { ArrowLeft, Zap, Lock, BarChart2, List, PieChart } from 'lucide-react';
import SubscriptionModal from '../components/SubscriptionModal';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import type { StrategyInfo, YearlyData } from '../services/types';

const StrategyDetail: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const [selectedYear, setSelectedYear] = useState<number | null>(null);
    const [isSubscribed, setIsSubscribed] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [strategyInfo, setStrategyInfo] = useState<StrategyInfo | null>(null);
    const [yearlyData, setYearlyData] = useState<YearlyData[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!id) return;

        const fetchData = async () => {
            try {
                setLoading(true);
                const [infoData, yearlyDataResponse] = await Promise.all([
                    api.getStrategyInfo(id),
                    api.getYearlyData(id)
                ]);
                setStrategyInfo(infoData);
                setYearlyData(yearlyDataResponse);
                // Set default selected year to current year if available
                const currentYear = new Date().getFullYear();
                const hasCurrentYear = yearlyDataResponse.some(y => y.year === currentYear);
                setSelectedYear(hasCurrentYear ? currentYear : null);
            } catch (error) {
                console.error('Failed to fetch strategy data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [id]);

    const handleSubscribe = () => {
        setIsSubscribed(true);
    };

    const formatPercent = (value: number) => {
        const sign = value >= 0 ? '+' : '';
        return `${sign}${(value * 100).toFixed(2)}%`;
    };

    const formatTotalReturn = (value: number) => {
        const sign = value >= 0 ? '+' : '';
        return `${sign}${(value * 100).toFixed(1)}%`;
    };

    if (loading || !strategyInfo) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', backgroundColor: 'var(--bg-main)' }}>
                <div style={{ color: 'var(--text-secondary)' }}>加载中...</div>
            </div>
        );
    }

    // Calculate total return from start (from equity data)
    const totalReturn = strategyInfo.total_metrics.total_return;

    return (
        <div style={{ display: 'flex', minHeight: '100vh' }}>
            {/* Zone L: Left Sidebar (Time Machine) */}
            <div style={{ width: '240px', backgroundColor: 'var(--bg-main)', borderRight: '1px solid var(--border-light)', padding: '24px', flexShrink: 0 }}>
                <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', marginBottom: '40px' }}>
                    <ArrowLeft size={16} /> 返回列表
                </Link>

                {/* Total Stats */}
                <div style={{ marginBottom: '40px', padding: '16px', background: 'var(--bg-card)', borderRadius: '8px' }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>总 (全周期)</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--color-up-red)', fontFamily: 'DIN Condensed' }}>
                        {formatTotalReturn(totalReturn)}
                    </div>
                </div>

                {/* Year List */}
                <h4 style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '16px', textTransform: 'uppercase' }}>Time Machine</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {yearlyData.map(item => (
                        <div
                            key={item.year}
                            onClick={() => setSelectedYear(item.year)}
                            style={{
                                padding: '12px 16px', borderRadius: '6px', cursor: 'pointer',
                                backgroundColor: selectedYear === item.year ? 'rgba(24, 144, 255, 0.1)' : 'transparent',
                                color: selectedYear === item.year ? 'var(--color-brand-blue)' : 'var(--text-secondary)',
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                            }}
                        >
                            <span>{item.year}</span>
                            {item.is_running ? (
                                <span style={{ fontSize: '10px', background: 'rgba(82, 196, 26, 0.2)', color: 'var(--color-down-green)', padding: '2px 6px', borderRadius: '4px' }}>RUNNING</span>
                            ) : (
                                <span style={{ fontSize: '12px', color: item.ret >= 0 ? 'var(--color-up-red)' : 'var(--color-down-green)' }}>
                                    {formatPercent(item.ret)}
                                </span>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Zone R: Right Content */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                {/* R1: Global Header */}
                <div style={{ padding: '40px 40px 20px', borderBottom: '1px solid var(--border-light)', backgroundColor: 'var(--bg-main)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                                <h1 style={{ fontSize: '32px' }}>{strategyInfo.name}</h1>
                                {strategyInfo.is_active !== false ? (
                                    <span style={{
                                        border: '1px solid var(--color-down-green)', color: 'var(--color-down-green)',
                                        padding: '2px 8px', borderRadius: '4px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px'
                                    }}>
                                        <div style={{ width: '6px', height: '6px', background: 'currentColor', borderRadius: '50%' }}></div>
                                        实盘运行中
                                    </span>
                                ) : (
                                    <span style={{
                                        border: '1px solid var(--text-secondary)', color: 'var(--text-secondary)',
                                        padding: '2px 8px', borderRadius: '4px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px'
                                    }}>
                                        <div style={{ width: '6px', height: '6px', background: 'currentColor', borderRadius: '50%' }}></div>
                                        已停止
                                    </span>
                                )}
                            </div>
                            <p style={{ color: 'var(--text-secondary)' }}>
                                {strategyInfo.description} • {strategyInfo.tags.map(tag => `[${tag}]`).join(' ')}
                            </p>
                        </div>

                        {/* Signal Console */}
                        <div style={{
                            background: 'rgba(24, 144, 255, 0.1)', border: '1px solid rgba(24, 144, 255, 0.2)',
                            borderRadius: '8px', padding: '16px 24px', display: 'flex', alignItems: 'center', gap: '24px'
                        }}>
                            <div>
                                <div style={{ fontWeight: 'bold', color: 'white', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <Zap size={16} className="text-gold" />
                                    {isSubscribed ? "实时信号已连接" : "实时信号直推"}
                                </div>
                                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>支持 微信/短信/App 推送</div>
                            </div>
                            {!isSubscribed && (
                                <button
                                    onClick={() => setIsModalOpen(true)}
                                    style={{
                                        background: 'var(--color-brand-blue)', color: 'white', border: 'none',
                                        display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 20px'
                                    }}
                                >
                                    <Lock size={16} /> 订阅解锁
                                </button>
                            )}
                        </div>
                    </div>

                    {/* KPI Strip */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', marginTop: '40px', gap: '40px' }}>
                        <div>
                            <div style={{ fontSize: '36px', fontWeight: 'bold', color: 'var(--color-up-red)', fontFamily: 'DIN Condensed' }}>
                                {formatPercent(strategyInfo.total_metrics.cagr)}
                            </div>
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>年化收益 (CAGR)</div>
                        </div>
                        <div>
                            <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                                {formatPercent(strategyInfo.total_metrics.win_rate)}
                            </div>
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>胜率 (Win Rate)</div>
                        </div>
                        <div>
                            <div style={{ fontSize: '24px', fontWeight: 'bold', color: 'var(--color-down-green)' }}>
                                {formatPercent(strategyInfo.total_metrics.max_drawdown)}
                            </div>
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>最大回撤</div>
                        </div>
                        <div>
                            <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                                {strategyInfo.total_metrics.sharpe.toFixed(2)}
                            </div>
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>夏普比率</div>
                        </div>
                    </div>
                </div>

                {/* R2: Tab Bar */}
                <div style={{ padding: '0 40px', borderBottom: '1px solid var(--border-light)' }}>
                    <div style={{ display: 'flex', gap: '40px' }}>
                        {[
                            { path: 'metrics', label: '数据指标', icon: BarChart2 },
                            { path: 'transactions', label: '交易记录', icon: List },
                            { path: 'holdings', label: '每日持仓', icon: PieChart },
                        ].map(tab => (
                            <NavLink
                                key={tab.path}
                                to={tab.path}
                                className={({ isActive }) => isActive ? "active-tab" : ""}
                                style={({ isActive }) => ({
                                    display: 'flex', alignItems: 'center', gap: '8px',
                                    padding: '20px 0', borderBottom: isActive ? '2px solid var(--color-brand-blue)' : '2px solid transparent',
                                    color: isActive ? 'var(--color-brand-blue)' : 'var(--text-secondary)',
                                    fontWeight: isActive ? 500 : 400
                                })}
                            >
                                <tab.icon size={18} /> {tab.label}
                            </NavLink>
                        ))}
                    </div>
                </div>

                {/* R3: Tab Content (Outlet) */}
                <div style={{ flex: 1, padding: '40px', overflowY: 'auto', backgroundColor: 'var(--bg-main)' }}>
                    <Outlet context={{ isSubscribed, requestSubscribe: () => setIsModalOpen(true), selectedYear }} />
                </div>
            </div>

            <SubscriptionModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onSubscribe={handleSubscribe}
            />
        </div>
    );
};

export default StrategyDetail;
