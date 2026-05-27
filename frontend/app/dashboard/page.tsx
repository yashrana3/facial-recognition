'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Filler, BarElement } from 'chart.js';
import { Doughnut, Line, Bar } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Filler, BarElement);

const BACKEND_URL = 'http://localhost:5001';

interface SessionInfo {
    session_id: string;
    total_frames: number;
    avg_engagement: number;
    start_time: string;
    end_time: string;
    dominant_state: string;
}

interface HourlyReport {
    id: number;
    report_start_time: string;
    report_end_time: string;
    session_id: string;
    avg_engagement: number;
    dominant_emotion: string;
    top_understanding_state: string;
    total_frames: number;
}

interface DashStats {
    session_id: string;
    elapsed_seconds: number;
    total_frames: number;
    state_counts: Record<string, number>;
    current_state: string;
    blink_rate: number;
    engagement_index: number;
    head_stable: boolean;
}

const STATE_COLORS: Record<string, string> = {
    Understanding: '#22c55e',
    Confused: '#f59e0b',
    Lost: '#ef4444',
    'No One Available': '#64748b',
};

const STATE_EMOJI: Record<string, string> = {
    Understanding: '✅',
    Confused: '😕',
    Lost: '💤',
    'No One Available': '🚫',
};

export default function DashboardPage() {
    const [stats, setStats] = useState<DashStats | null>(null);
    const [sessions, setSessions] = useState<SessionInfo[]>([]);
    const [hourlyReports, setHourlyReports] = useState<HourlyReport[]>([]);
    const [engagementTimeline, setEngagementTimeline] = useState<{ time: string; value: number }[]>([]);
    const [loading, setLoading] = useState(true);
    const [autoRefresh, setAutoRefresh] = useState(true);

    const fetchAll = async () => {
        try {
            const [statsRes, sessRes, hourlyRes] = await Promise.all([
                fetch(`${BACKEND_URL}/api/dashboard/stats`),
                fetch(`${BACKEND_URL}/api/dashboard/sessions`),
                fetch(`${BACKEND_URL}/api/dashboard/hourly-reports`),
            ]);
            if (statsRes.ok) {
                const s = await statsRes.json();
                setStats(s);
                // Append to engagement timeline
                const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
                setEngagementTimeline(prev => [...prev, { time: timeStr, value: s.engagement_index }].slice(-60));
            }
            if (sessRes.ok) setSessions(await sessRes.json());
            if (hourlyRes.ok) setHourlyReports(await hourlyRes.json());
        } catch (e) { console.error('Dashboard fetch error:', e); }
        setLoading(false);
    };

    useEffect(() => {
        fetchAll();
        if (!autoRefresh) return;
        const iv = setInterval(fetchAll, 5000);
        return () => clearInterval(iv);
    }, [autoRefresh]);

    const formatElapsed = (s: number) => {
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = Math.floor(s % 60);
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
    };

    // ── Chart: State Distribution Doughnut ──
    const stateCounts = stats?.state_counts || {};
    const totalStateCounts = Object.values(stateCounts).reduce((a, b) => a + b, 0);
    const doughnutData = {
        labels: ['Understanding', 'Confused', 'Lost'],
        datasets: [{
            data: [stateCounts['Understanding'] || 0, stateCounts['Confused'] || 0, stateCounts['Lost'] || 0],
            backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
            borderColor: ['#16a34a', '#d97706', '#dc2626'],
            borderWidth: 2,
            hoverOffset: 8,
        }],
    };
    const doughnutOptions = {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
            legend: { position: 'bottom' as const, labels: { color: '#aaa', font: { size: 11 }, padding: 12, usePointStyle: true, pointStyleWidth: 8 } },
            tooltip: { backgroundColor: '#1e1e2e', titleColor: '#ccc', bodyColor: '#aaa' },
        },
    };

    // ── Chart: Engagement Line ──
    const lineData = {
        labels: engagementTimeline.map(e => e.time),
        datasets: [{
            label: 'Engagement %',
            data: engagementTimeline.map(e => Math.round(e.value * 100)),
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.08)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.4,
            fill: true,
        }],
    };
    const lineOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { min: 0, max: 100, ticks: { color: '#555', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
            x: { ticks: { color: '#555', font: { size: 9 }, maxTicksLimit: 10 }, grid: { display: false } },
        },
        plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: '#1e1e2e', titleColor: '#ccc', bodyColor: '#aaa' },
        },
    };

    // ── Chart: Session Engagement Bar ──
    const recentSessions = sessions.slice(0, 10).reverse();
    const barData = {
        labels: recentSessions.map(s => s.session_id?.slice(0, 6)),
        datasets: [{
            label: 'Avg Engagement %',
            data: recentSessions.map(s => Math.round((s.avg_engagement || 0) * 100)),
            backgroundColor: recentSessions.map(s => {
                const e = s.avg_engagement || 0;
                return e > 0.6 ? 'rgba(34,197,94,0.6)' : e > 0.3 ? 'rgba(245,158,11,0.6)' : 'rgba(239,68,68,0.6)';
            }),
            borderRadius: 6,
            borderSkipped: false,
        }],
    };
    const barOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { min: 0, max: 100, ticks: { color: '#555', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
            x: { ticks: { color: '#555', font: { size: 9 } }, grid: { display: false } },
        },
        plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: '#1e1e2e', titleColor: '#ccc', bodyColor: '#aaa' },
        },
    };

    const currentColor = STATE_COLORS[stats?.current_state || 'No One Available'] || '#64748b';
    const currentEmoji = STATE_EMOJI[stats?.current_state || 'No One Available'] || '❓';

    return (
        <div className="dashboard-container">
            {/* ─── Nav Bar ─── */}
            <nav className="nav-bar">
                <div className="nav-left">
                    {/* <span className="nav-logo">📊</span> */}
                    <span className="nav-brand">Attention Dashboard</span>
                </div>
                <div className="nav-links">
                    <Link href="/" className="nav-link">Camera</Link>
                    <Link href="/dashboard" className="nav-link nav-link-active">Dashboard</Link>
                </div>
            </nav>

            {/* ─── Header ─── */}
            <header className="dash-header">
                <div className="dash-header-left">
                    <h1 className="dash-title">Analytics Dashboard</h1>
                    <span className="dash-subtitle">Session data & engagement graphs</span>
                </div>
                <div className="dash-header-right">
                    <button
                        className={`refresh-toggle ${autoRefresh ? 'refresh-on' : 'refresh-off'}`}
                        onClick={() => setAutoRefresh(!autoRefresh)}
                    >
                        {autoRefresh ? '🔄 Auto-Refresh ON' : '⏸ Auto-Refresh OFF'}
                    </button>
                    {stats && (
                        <div className="session-timer">
                            <span className="timer-label">Session</span>
                            <span className="timer-value">{formatElapsed(stats.elapsed_seconds)}</span>
                        </div>
                    )}
                </div>
            </header>

            {loading ? (
                <div className="loading-state">
                    <span className="loading-spinner" />
                    <span>Loading dashboard data...</span>
                </div>
            ) : (
                <>
                    {/* ───── LIVE STATS ROW ───── */}
                    <div className="stats-row">
                        <div className="stat-card" style={{ borderTopColor: currentColor }}>
                            <div className="stat-card-icon" style={{ color: currentColor }}>{currentEmoji}</div>
                            <div className="stat-card-body">
                                <span className="stat-card-label">Current State</span>
                                <span className="stat-card-value" style={{ color: currentColor }}>{stats?.current_state || '—'}</span>
                            </div>
                        </div>
                        <div className="stat-card" style={{ borderTopColor: '#8b5cf6' }}>
                            <div className="stat-card-icon" style={{ color: '#8b5cf6' }}>📈</div>
                            <div className="stat-card-body">
                                <span className="stat-card-label">Engagement</span>
                                <span className="stat-card-value">{stats ? (stats.engagement_index * 100).toFixed(0) + '%' : '—'}</span>
                            </div>
                        </div>
                        <div className="stat-card" style={{ borderTopColor: '#3b82f6' }}>
                            <div className="stat-card-icon" style={{ color: '#3b82f6' }}>👁</div>
                            <div className="stat-card-body">
                                <span className="stat-card-label">Blink Rate</span>
                                <span className="stat-card-value">{stats?.blink_rate || 0} /min</span>
                            </div>
                        </div>
                        <div className="stat-card" style={{ borderTopColor: '#06b6d4' }}>
                            <div className="stat-card-icon" style={{ color: '#06b6d4' }}>🎯</div>
                            <div className="stat-card-body">
                                <span className="stat-card-label">Frames Analyzed</span>
                                <span className="stat-card-value">{stats?.total_frames || 0}</span>
                            </div>
                        </div>
                    </div>

                    {/* ───── CHARTS ROW 1 ───── */}
                    <div className="charts-row">
                        <div className="chart-card chart-card-small">
                            <div className="chart-card-header">
                                <span>🎯 State Distribution</span>
                                <span className="chart-card-sub">{totalStateCounts} readings</span>
                            </div>
                            <div className="chart-content doughnut-wrap">
                                {totalStateCounts > 0 ? (
                                    <Doughnut data={doughnutData} options={doughnutOptions} />
                                ) : (
                                    <div className="chart-empty">
                                        <span>No data yet</span>
                                        <span className="chart-empty-sub">Open the Camera page to start recording</span>
                                    </div>
                                )}
                            </div>
                            {totalStateCounts > 0 && (
                                <div className="state-pct-row">
                                    {['Understanding', 'Confused', 'Lost'].map(state => {
                                        const count = stateCounts[state] || 0;
                                        const pct = totalStateCounts > 0 ? (count / totalStateCounts * 100).toFixed(0) : '0';
                                        return (
                                            <div key={state} className="state-pct-item">
                                                <div className="state-pct-bar-track">
                                                    <div className="state-pct-bar-fill" style={{ width: `${pct}%`, background: STATE_COLORS[state] }} />
                                                </div>
                                                <span className="state-pct-label" style={{ color: STATE_COLORS[state] }}>{STATE_EMOJI[state]} {pct}%</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>

                        <div className="chart-card chart-card-large">
                            <div className="chart-card-header">
                                <span>📈 Engagement Over Time</span>
                                <span className="chart-card-sub">Live — {engagementTimeline.length} points</span>
                            </div>
                            <div className="chart-content line-wrap">
                                {engagementTimeline.length > 1 ? (
                                    <Line data={lineData} options={lineOptions} />
                                ) : (
                                    <div className="chart-empty">
                                        <span>Waiting for data...</span>
                                        <span className="chart-empty-sub">Chart populates as the session runs</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* ───── CHARTS ROW 2: Session Bar Chart ───── */}
                    {recentSessions.length > 0 && (
                        <div className="charts-row charts-row-full">
                            <div className="chart-card chart-card-full">
                                <div className="chart-card-header">
                                    <span>📊 Session Engagement Comparison</span>
                                    <span className="chart-card-sub">Last {recentSessions.length} sessions</span>
                                </div>
                                <div className="chart-content bar-wrap">
                                    <Bar data={barData} options={barOptions} />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* ───── TABLES ───── */}
                    <div className="bottom-grid">
                        <div className="table-card">
                            <div className="table-card-header">
                                <span>⏱ Hourly Reports</span>
                                <span className="table-card-count">{hourlyReports.length}</span>
                            </div>
                            <div className="table-scroll">
                                {hourlyReports.length > 0 ? (
                                    <table className="dash-table">
                                        <thead>
                                            <tr>
                                                <th>Time</th>
                                                <th>Emotion</th>
                                                <th>State</th>
                                                <th>Engagement</th>
                                                <th>Frames</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {hourlyReports.map((r, i) => (
                                                <tr key={i}>
                                                    <td className="td-time">{r.report_end_time ? new Date(r.report_end_time).toLocaleTimeString() : '—'}</td>
                                                    <td style={{ textTransform: 'capitalize' }}>{r.dominant_emotion}</td>
                                                    <td>
                                                        <span className="td-state" style={{ color: STATE_COLORS[r.top_understanding_state] }}>
                                                            {STATE_EMOJI[r.top_understanding_state]} {r.top_understanding_state}
                                                        </span>
                                                    </td>
                                                    <td>{r.avg_engagement != null ? (r.avg_engagement * 100).toFixed(0) + '%' : '—'}</td>
                                                    <td>{r.total_frames}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                ) : (
                                    <div className="table-empty">No hourly reports yet. Reports are generated every hour.</div>
                                )}
                            </div>
                        </div>

                        <div className="table-card">
                            <div className="table-card-header">
                                <span>📋 Session History</span>
                                <span className="table-card-count">{sessions.length}</span>
                            </div>
                            <div className="table-scroll">
                                {sessions.length > 0 ? (
                                    <table className="dash-table">
                                        <thead>
                                            <tr>
                                                <th>Session</th>
                                                <th>Start</th>
                                                <th>State</th>
                                                <th>Engagement</th>
                                                <th>Frames</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {sessions.map((s, i) => (
                                                <tr key={i}>
                                                    <td className="td-session-id">{s.session_id?.slice(0, 8)}...</td>
                                                    <td className="td-time">{s.start_time ? new Date(s.start_time).toLocaleString() : '—'}</td>
                                                    <td>
                                                        <span className="td-state" style={{ color: STATE_COLORS[s.dominant_state] }}>
                                                            {STATE_EMOJI[s.dominant_state]} {s.dominant_state}
                                                        </span>
                                                    </td>
                                                    <td>{s.avg_engagement != null ? (s.avg_engagement * 100).toFixed(0) + '%' : '—'}</td>
                                                    <td>{s.total_frames}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                ) : (
                                    <div className="table-empty">No sessions recorded yet.</div>
                                )}
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
