import React, { useEffect, useState, useCallback } from 'react';
import { Search, Bell, Activity, Flame, Shield, Zap } from 'lucide-react';
import type { StrategyListItem, StrategyListParams } from '../services/types';
import { api } from '../services/api';
import StrategyCard from '../components/StrategyCard';

type SortField = 'cagr' | 'sharpe' | 'max_drawdown' | 'win_rate' | 'latest_signal';

const Marketplace: React.FC = () => {
    const [strategies, setStrategies] = useState<StrategyListItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [sortBy, setSortBy] = useState<SortField>('cagr');
    const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

    const fetchStrategies = useCallback(async () => {
        setLoading(true);
        try {
            const params: StrategyListParams = {
                search: searchQuery || undefined,
                sort: sortBy,
                order: sortOrder
            };
            const data = await api.getStrategies(params);
            setStrategies(data);
        } catch (err) {
            console.error("Failed to fetch strategies", err);
        } finally {
            setLoading(false);
        }
    }, [searchQuery, sortBy, sortOrder]);

    useEffect(() => {
        fetchStrategies();
    }, [fetchStrategies]);

    const handleSort = (field: SortField) => {
        if (sortBy === field) {
            // Toggle order if clicking same field
            setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
        } else {
            setSortBy(field);
            setSortOrder('desc'); // Default to desc for new field
        }
    };

    const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSearchQuery(e.target.value);
    };

    const getSortButtonStyle = (field: SortField) => {
        const isActive = sortBy === field;
        return {
            background: isActive ? 'rgba(24, 144, 255, 0.2)' : 'transparent',
            color: isActive ? 'var(--color-brand-blue)' : 'var(--text-secondary)',
            border: isActive ? '1px solid var(--color-brand-blue)' : '1px solid transparent',
            borderRadius: '4px',
            padding: '6px 12px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            fontSize: '14px',
            transition: 'all 0.2s'
        };
    };

    return (
        <div className="container" style={{ paddingBottom: '40px' }}>
            {/* Zone A: Hero Header */}
            <div style={{ padding: '60px 0 40px', textAlign: 'center' }}>
                <h1 style={{ fontSize: '48px', fontWeight: 'bold', marginBottom: '16px' }}>量化信号跟投中心</h1>
                <p style={{ color: 'var(--text-secondary)', fontSize: '16px', marginBottom: '40px' }}>
                    别再凭感觉瞎买。订阅经过实盘验证的量化策略，自动接收买卖信号。
                </p>

                {/* Onboarding Steps */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '24px', maxWidth: '900px', margin: '0 auto' }}>
                    {[
                        { icon: Search, title: '挑选策略', desc: '按年化收益筛选' },
                        { icon: Bell, title: '订阅信号', desc: '微信/短信实时接收' },
                        { icon: Activity, title: '跟着买卖', desc: '傻瓜式跟单获利' }
                    ].map((step, i) => (
                        <div key={i} style={{
                            background: 'rgba(255,255,255,0.03)', padding: '24px', borderRadius: '12px',
                            backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.05)',
                            display: 'flex', flexDirection: 'column', alignItems: 'center'
                        }}>
                            <div style={{ background: 'var(--bg-main)', padding: '12px', borderRadius: '50%', marginBottom: '16px' }}>
                                <step.icon className="text-blue" size={24} />
                            </div>
                            <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>{step.title}</h3>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '14px', margin: 0 }}>{step.desc}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Zone B: Filter Bar */}
            <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                marginBottom: '24px', backgroundColor: 'var(--bg-card)', padding: '16px 24px', borderRadius: '8px'
            }}>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                        onClick={() => handleSort('cagr')}
                        style={getSortButtonStyle('cagr')}
                    >
                        <Flame size={14} /> 年化收益 {sortBy === 'cagr' ? (sortOrder === 'desc' ? '↓' : '↑') : ''}
                    </button>
                    <button
                        onClick={() => handleSort('max_drawdown')}
                        style={getSortButtonStyle('max_drawdown')}
                    >
                        <Shield size={14} /> 最大回撤 {sortBy === 'max_drawdown' ? (sortOrder === 'desc' ? '↓' : '↑') : ''}
                    </button>
                    <button
                        onClick={() => handleSort('win_rate')}
                        style={getSortButtonStyle('win_rate')}
                    >
                        胜率 {sortBy === 'win_rate' ? (sortOrder === 'desc' ? '↓' : '↑') : ''}
                    </button>
                    <button
                        onClick={() => handleSort('sharpe')}
                        style={getSortButtonStyle('sharpe')}
                    >
                        夏普比率 {sortBy === 'sharpe' ? (sortOrder === 'desc' ? '↓' : '↑') : ''}
                    </button>
                </div>
                <div style={{ position: 'relative' }}>
                    <Search style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} size={16} />
                    <input
                        type="text"
                        placeholder="输入策略名称或标签搜索..."
                        value={searchQuery}
                        onChange={handleSearch}
                        style={{
                            background: 'var(--bg-main)', border: '1px solid var(--border-light)',
                            borderRadius: '100px', padding: '8px 16px 8px 36px', color: 'white', outline: 'none', minWidth: '240px'
                        }}
                    />
                </div>
            </div>

            {/* Zone C: Strategy List */}
            <div>
                {loading ? (
                    <div className="text-center" style={{ padding: '40px', color: 'var(--text-secondary)' }}>
                        正在加载策略数据...
                    </div>
                ) : strategies.length === 0 ? (
                    <div className="text-center" style={{ padding: '40px', color: 'var(--text-secondary)' }}>
                        没有找到匹配的策略
                    </div>
                ) : (
                    strategies.map(strategy => (
                        <StrategyCard key={strategy.id} strategy={strategy} />
                    ))
                )}
            </div>
        </div>
    );
};

export default Marketplace;
