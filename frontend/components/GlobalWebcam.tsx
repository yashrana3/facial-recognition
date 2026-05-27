'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import io from 'socket.io-client';

const BACKEND_URL = 'http://localhost:5001';
const NO_FACE_POPUP_DELAY_MS = 5000; // show popup after 5s of no face

interface AnalysisResult {
    face_detected: boolean;
    emotion: string;
    understanding: string;
    engagement_index: number;
    blink_rate: number;
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

const STATE_BG: Record<string, string> = {
    Understanding: 'rgba(34,197,94,0.12)',
    Confused: 'rgba(245,158,11,0.12)',
    Lost: 'rgba(239,68,68,0.12)',
    'No One Available': 'rgba(100,116,139,0.12)',
};

const mapToThreeStates = (state: string): string => {
    switch (state) {
        case 'Understanding': return 'Understanding';
        case 'Confused': return 'Confused';
        case 'Frustrated': return 'Confused';
        case 'Lost': return 'Lost';
        case 'Distracted': return 'Lost';
        case 'Unclear': return 'Confused';
        default: return state;
    }
};

export default function GlobalWebcam() {
    const webcamRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const socketRef = useRef<any>(null);
    const noFaceTimerRef = useRef<NodeJS.Timeout | null>(null);

    const [data, setData] = useState<AnalysisResult>({
        face_detected: false,
        emotion: 'neutral',
        understanding: 'No One Available',
        engagement_index: 0,
        blink_rate: 0,
    });
    const [isWebcamActive, setIsWebcamActive] = useState(false);
    const [confirmedState, setConfirmedState] = useState('No One Available');
    const [isConnected, setIsConnected] = useState(false);

    // ── No-face popup state ──
    const [showNoFacePopup, setShowNoFacePopup] = useState(false);
    const [noFaceReason, setNoFaceReason] = useState('');
    const [submittedReasons, setSubmittedReasons] = useState<{ time: string; reason: string }[]>([]);

    // ── Webcam init ──
    useEffect(() => {
        const init = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                if (webcamRef.current) { webcamRef.current.srcObject = stream; setIsWebcamActive(true); }
            } catch (e) { console.error('Webcam error:', e); }
        };
        init();
        return () => {
            if (webcamRef.current?.srcObject)
                (webcamRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        };
    }, []);

    // ── Socket.IO ──
    useEffect(() => {
        socketRef.current = io(BACKEND_URL);

        socketRef.current.on('connect', () => setIsConnected(true));
        socketRef.current.on('disconnect', () => setIsConnected(false));

        socketRef.current.on('emotion_result', (res: any) => {
            // Use backend-provided confirmed state if available, otherwise raw state
            const state = res.confirmed_state || mapToThreeStates(res.understanding);
            setConfirmedState(state);
            setData(res);
        });

        return () => { socketRef.current?.disconnect(); };
    }, []);

    // ── No-face popup timer logic ──
    useEffect(() => {
        if (!isConnected || !isWebcamActive) return;

        if (!data.face_detected) {
            // Start timer if not already running
            if (!noFaceTimerRef.current) {
                noFaceTimerRef.current = setTimeout(() => {
                    setShowNoFacePopup(true);
                }, NO_FACE_POPUP_DELAY_MS);
            }
        } else {
            // Face detected — clear timer and close popup
            if (noFaceTimerRef.current) {
                clearTimeout(noFaceTimerRef.current);
                noFaceTimerRef.current = null;
            }
            setShowNoFacePopup(false);
        }

        return () => {
            if (noFaceTimerRef.current) {
                clearTimeout(noFaceTimerRef.current);
                noFaceTimerRef.current = null;
            }
        };
    }, [data.face_detected, isConnected, isWebcamActive]);

    // ── Handle reason submission ──
    const handleReasonSubmit = useCallback(() => {
        const trimmed = noFaceReason.trim();
        if (!trimmed) return;

        const entry = {
            time: new Date().toLocaleTimeString(),
            reason: trimmed,
        };
        setSubmittedReasons(prev => [...prev, entry]);

        // Optionally send to backend
        if (socketRef.current?.connected) {
            socketRef.current.emit('no_face_reason', entry);
        }

        setNoFaceReason('');
        setShowNoFacePopup(false);

        // Reset timer so popup can appear again if face still not detected
        noFaceTimerRef.current = setTimeout(() => {
            setShowNoFacePopup(true);
        }, NO_FACE_POPUP_DELAY_MS);
    }, [noFaceReason]);

    const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
        if (e.key === 'Enter') handleReasonSubmit();
        if (e.key === 'Escape') setShowNoFacePopup(false);
    }, [handleReasonSubmit]);

    const cfg = STATE_COLORS[confirmedState] || STATE_COLORS['No One Available'];
    const emoji = STATE_EMOJI[confirmedState] || '❓';
    const bgColor = STATE_BG[confirmedState] || STATE_BG['No One Available'];

    return (
        <>
            {/* ─── No-Face Popup Modal ─── */}
            {showNoFacePopup && (
                <div className="nf-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowNoFacePopup(false); }}>
                    <div className="nf-modal">
                        <div className="nf-modal-icon">🚫</div>
                        <h2 className="nf-modal-title">No Face Detected</h2>
                        <p className="nf-modal-desc">
                            We can't see your face in the camera. Please let us know why so we can log it for your session report.
                        </p>
                        <div className="nf-input-wrap">
                            <label className="nf-label" htmlFor="nf-reason-input">
                                Why is your face not visible?
                            </label>
                            <input
                                id="nf-reason-input"
                                className="nf-input"
                                type="text"
                                placeholder="e.g. Taking a break, looking at notes, phone call..."
                                value={noFaceReason}
                                onChange={e => setNoFaceReason(e.target.value)}
                                onKeyDown={handleKeyDown}
                                autoFocus
                            />
                        </div>
                        <div className="nf-modal-actions">
                            <button
                                className="nf-btn nf-btn-primary"
                                onClick={handleReasonSubmit}
                                disabled={!noFaceReason.trim()}
                            >
                                Submit Reason
                            </button>
                            <button
                                className="nf-btn nf-btn-secondary"
                                onClick={() => setShowNoFacePopup(false)}
                            >
                                Dismiss
                            </button>
                        </div>
                        {submittedReasons.length > 0 && (
                            <div className="nf-history">
                                <span className="nf-history-title">Previous reasons this session:</span>
                                {submittedReasons.slice(-3).map((r, i) => (
                                    <div key={i} className="nf-history-item">
                                        <span className="nf-history-time">{r.time}</span>
                                        <span className="nf-history-reason">{r.reason}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ─── Camera Widget ─── */}
            <div className="webcam-widget">
                <div className="webcam-widget-header">
                    <span className="webcam-dot" style={{ background: isConnected && isWebcamActive ? '#22c55e' : '#ef4444' }} />
                    <span className="webcam-widget-title">Live Camera</span>
                </div>
                <div className="webcam-feed-wrapper">
                    <video ref={webcamRef} autoPlay muted playsInline className="webcam-feed" />
                    <canvas ref={canvasRef} style={{ display: 'none' }} />

                    {/* Connection Status Overlay */}
                    {!isConnected && (
                        <div className="no-face-overlay" style={{ background: 'rgba(0,0,0,0.85)', zIndex: 10 }}>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{ fontSize: '24px', marginBottom: '8px' }}>🔌</div>
                                <span>Connecting to server...</span>
                            </div>
                        </div>
                    )}

                    {/* No Face Overlay */}
                    {isConnected && !data.face_detected && isWebcamActive && (
                        <div className="no-face-overlay">
                            <span>🚫 No face</span>
                        </div>
                    )}
                </div>
                <div className="webcam-state-badge" style={{ background: bgColor, color: cfg, borderColor: cfg }}>
                    {emoji} {confirmedState}
                </div>
            </div>
        </>
    );
}
