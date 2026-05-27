'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function CameraPage() {
  const [sessionElapsed, setSessionElapsed] = useState(0);
  const [isTabVisible, setIsTabVisible] = useState(true);
  const [wakeLock, setWakeLock] = useState<any>(null);

  // ── Page Visibility & Wake Lock (keep running in background) ──
  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsTabVisible(!document.hidden);
      if (!document.hidden) {
        console.log('Tab is now visible - monitoring continues');
      } else {
        console.log('Tab is hidden - monitoring continues in background');
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    const requestWakeLock = async () => {
      try {
        if ('wakeLock' in navigator) {
          const lock = await (navigator as any).wakeLock.request('screen');
          setWakeLock(lock);
          console.log('Wake Lock acquired - system will stay awake');

          lock.addEventListener('release', () => {
            console.log('Wake Lock released');
          });
        }
      } catch (err) {
        console.warn('Wake Lock not supported or denied:', err);
      }
    };

    requestWakeLock();

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (wakeLock) {
        wakeLock.release();
      }
    };
  }, [wakeLock]);

  // ── Timer ──
  useEffect(() => {
    const iv = setInterval(() => setSessionElapsed(prev => prev + 1), 1000);
    return () => clearInterval(iv);
  }, []);

  const formatElapsed = (s: number) => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="home-page">
      {/* ═══ BACKGROUND MONITORING INDICATOR ═══ */}
      {!isTabVisible && (
        <div className="bg-monitor-banner">
          <span className="bg-monitor-icon">👁️</span>
          <span className="bg-monitor-text">
            <strong>Background Monitoring Active</strong> — Camera and analysis continue running while you use other apps
          </span>
        </div>
      )}

      {/* ─── Main Content ─── */}
      <div className="home-container">
        <header className="home-header">
          <div className="home-header-left">
            <h1 className="home-title">Attention Monitor</h1>
            <p className="home-subtitle">Facial expression analysis running in the background</p>
          </div>
          <div className="home-header-right">
            <div className="session-timer">
              <span className="timer-label">Session</span>
              <span className="timer-value">{formatElapsed(sessionElapsed)}</span>
            </div>
          </div>
        </header>

        <div className="home-content">
          <div className="info-card">
            {/* <div className="info-icon">🎯</div> */}
            <h2 className="info-title">Background Monitoring Active</h2>
            <p className="info-desc">
              Your facial expressions are being analyzed in real-time. The camera and analysis <strong>continue running even when you switch to other apps or tabs</strong>. The small camera window in the top-right corner shows your current state.
            </p>
            {/* <div className="info-note">
              <strong>💡 Tip:</strong> The camera widget stays visible on all pages, including the dashboard. Your session continues uninterrupted when you navigate between pages.
            </div> */}
          </div>

          <div className="action-card">
            {/* <div className="action-icon">📊</div> */}
            <h2 className="action-title">View Analytics Dashboard</h2>
            <p className="action-desc">
              See detailed charts, graphs, and reports of your attention patterns, engagement levels, and session history. The camera continues running while you view the dashboard.
            </p>
            <Link href="/dashboard" className="dashboard-link-btn">
              Open Dashboard →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
