import React, { useState } from 'react';
import { X, Check, Lock, Zap, Shield, Crown, Clock } from 'lucide-react';

interface SubscriptionModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSubscribe: () => void;
    strategyName?: string;
}

const SubscriptionModal: React.FC<SubscriptionModalProps> = ({ isOpen, onClose, onSubscribe, strategyName = "Alpha Trend Strategy" }) => {
    const [isLoading, setIsLoading] = useState(false);

    if (!isOpen) return null;

    const handleSubscribe = () => {
        setIsLoading(true);
        // Simulate payment delay
        setTimeout(() => {
            setIsLoading(false);
            onSubscribe();
            onClose();
        }, 1500);
    };

    return (
        <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'var(--bg-overlay)', backdropFilter: 'blur(4px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
        }}>
            <div className="modal-content" style={{
                backgroundColor: 'var(--bg-main)', borderRadius: '12px',
                width: '1080px', maxWidth: '95vw', maxHeight: '90vh', overflowY: 'auto',
                position: 'relative', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
            }}>
                {/* Close Button */}
                <button onClick={onClose} style={{
                    position: 'absolute', top: '24px', right: '24px',
                    background: 'transparent', border: 'none', color: 'var(--text-secondary)'
                }}>
                    <X size={24} />
                </button>

                <div style={{ padding: '40px' }}>
                    {/* Header */}
                    <div className="text-center" style={{ marginBottom: '40px' }}>
                        <h2 style={{ fontSize: '24px', marginBottom: '16px' }}>解锁 {strategyName} 核心权益</h2>
                        <div style={{
                            display: 'inline-flex', alignItems: 'center', gap: '8px',
                            background: 'rgba(245, 34, 45, 0.1)', color: 'var(--color-up-red)',
                            padding: '6px 16px', borderRadius: '100px', fontSize: '12px'
                        }}>
                            <Clock size={14} />
                            <span>距下一笔交易信号触发可能不到 2 小时</span>
                        </div>
                    </div>

                    {/* Pricing Cards */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.1fr 1fr', gap: '24px', alignItems: 'center' }}>

                        {/* Free Tier */}
                        <div style={{
                            border: '1px solid var(--border-light)', borderRadius: '8px', padding: '32px',
                            opacity: 0.7
                        }}>
                            <h3 style={{ fontSize: '18px', color: 'var(--text-secondary)' }}>游客 (Free)</h3>
                            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '24px' }}>适合观望考察</p>
                            <div style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '24px' }}>¥ 0 <span style={{ fontSize: '14px', fontWeight: 'normal' }}>/ 永久</span></div>
                            <button disabled style={{ width: '100%', marginBottom: '24px', cursor: 'not-allowed', color: 'var(--text-secondary)' }}>当前状态</button>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
                                <div style={{ display: 'flex', gap: '8px' }}><Check size={16} className="text-green" /> 查看历史净值</div>
                                <div style={{ display: 'flex', gap: '8px' }}><Check size={16} className="text-green" /> 验证已清仓记录</div>
                                <div style={{ display: 'flex', gap: '8px', opacity: 0.5 }}><X size={16} className="text-red" /> 当前持仓加密</div>
                                <div style={{ display: 'flex', gap: '8px', opacity: 0.5 }}><X size={16} className="text-red" /> 无信号通知</div>
                            </div>
                        </div>

                        {/* Plus Tier (Recommended) */}
                        <div style={{
                            border: '2px solid var(--color-brand-blue)', borderRadius: '12px', padding: '40px 32px',
                            backgroundColor: 'rgba(24, 144, 255, 0.05)', position: 'relative',
                            transform: 'scale(1.02)', boxShadow: '0 0 40px rgba(24, 144, 255, 0.1)'
                        }}>
                            <div style={{
                                position: 'absolute', top: 0, left: '50%', transform: 'translate(-50%, -50%)',
                                background: 'var(--color-brand-blue)', color: 'white',
                                padding: '4px 12px', borderRadius: '100px', fontSize: '12px', fontWeight: 'bold'
                            }}>RECOMMENDED</div>

                            <h3 style={{ fontSize: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                本策略订阅 (Plus) <Zap size={18} fill="currentColor" className="text-red" />
                            </h3>
                            <p style={{ fontSize: '12px', color: 'var(--color-brand-blue)', marginBottom: '24px' }}>适合实盘跟随</p>
                            <div style={{ fontSize: '48px', fontWeight: 'bold', marginBottom: '32px' }}>¥ 299 <span style={{ fontSize: '16px', fontWeight: 'normal', color: 'var(--text-secondary)' }}>/ 月</span></div>

                            <button onClick={handleSubscribe} className="bg-brand-blue" style={{
                                width: '100%', padding: '16px', background: 'var(--color-brand-blue)', color: 'white',
                                border: 'none', borderRadius: '8px', fontSize: '16px', fontWeight: 'bold',
                                marginBottom: '12px', cursor: isLoading ? 'wait' : 'pointer'
                            }}>
                                {isLoading ? '正在处理...' : '立即订阅'}
                            </button>
                            <div className="text-center text-gray" style={{ fontSize: '12px', marginBottom: '32px' }}>支持 7 天无理由退款</div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', fontSize: '14px' }}>
                                <div style={{ display: 'flex', gap: '8px' }}><Check size={18} className="text-blue" /> 包含免费版所有功能</div>
                                <div style={{ display: 'flex', gap: '8px', fontWeight: 'bold' }}><Lock size={18} className="text-blue" /> 解锁实时持仓代码</div>
                                <div style={{ display: 'flex', gap: '8px', fontWeight: 'bold' }}><Zap size={18} className="text-blue" /> 实时买卖信号推送</div>
                                <div style={{ display: 'flex', gap: '8px' }}><Check size={18} className="text-blue" /> 支持 微信/短信 通知</div>
                            </div>
                        </div>

                        {/* Pro Tier */}
                        <div style={{
                            border: '1px solid var(--border-light)', borderRadius: '8px', padding: '32px',
                        }}>
                            <h3 style={{ fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                PRO 全站通 <Crown size={16} className="text-gold" />
                            </h3>
                            <p style={{ fontSize: '12px', color: 'var(--color-gold)', marginBottom: '24px' }}>适合资产配置</p>
                            <div style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '24px' }}>¥ 699 <span style={{ fontSize: '14px', fontWeight: 'normal' }}>/ 月</span></div>
                            <button style={{
                                width: '100%', marginBottom: '24px', background: 'transparent',
                                border: '1px solid var(--text-primary)', color: 'var(--text-primary)'
                            }}>升级 PRO</button>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
                                <div style={{ display: 'flex', gap: '8px' }}><Check size={16} /> 包含单策略所有功能</div>
                                <div style={{ display: 'flex', gap: '8px' }}><Lock size={16} /> 解锁全站 15+ 个策略</div>
                                <div style={{ display: 'flex', gap: '8px' }}><Zap size={16} /> VIP 极速推送通道</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer Trust Badges */}
                <div style={{
                    backgroundColor: 'rgba(255,255,255,0.03)', padding: '24px 40px',
                    display: 'flex', justifyContent: 'space-around', borderTop: '1px solid var(--border-light)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <Shield className="text-blue" size={24} />
                        <div>
                            <div style={{ fontWeight: 'bold', fontSize: '14px' }}>7天无理由退款</div>
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>试用不满意？一键全额退款</div>
                        </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <Lock className="text-blue" size={24} />
                        <div>
                            <div style={{ fontWeight: 'bold', fontSize: '14px' }}>数据安全</div>
                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>SSL 加密支付保障</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SubscriptionModal;
