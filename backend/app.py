from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import base64
import os
import io
import time
import math
from collections import deque
from PIL import Image
import urllib.request
import uuid
from flask_apscheduler import APScheduler
from database import init_db, log_frame, aggregate_last_hour

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Scheduler Config
class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config())

# Initialize Scheduler
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Schedule Aggregation Job (Run every hour)
@scheduler.task('interval', id='hourly_aggregation', hours=1)
def scheduled_aggregation():
    with app.app_context():
        print("Scheduler: Running hourly aggregation for all active sessions...")
        from database import aggregate_all_active_sessions
        aggregate_all_active_sessions()

# ... (rest of imports and app setup)

# Video file path
VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'lecture_test_video.mp4')

# ────────────────────────────────────────
# MediaPipe Face Landmarker (new API)
# ────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded.")

# Create face landmarker
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ────────────────────────────────────────
# EAR (Eye Aspect Ratio) for blink detection
# ────────────────────────────────────────
# MediaPipe Face Mesh eye landmark indices
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # p1-p6
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # p1-p6

EAR_THRESHOLD = 0.21          # below this → eye is closed
EAR_CONSEC_FRAMES = 2         # min consecutive frames for a blink
LONG_CLOSURE_SEC = 1.5        # eye closed longer than this → distraction

# ────────────────────────────────────────
# Blink speed config  (eyelid velocity)
# ────────────────────────────────────────
BLINK_SPEED_NORMAL_MS = 150   # normal blink ~100-150 ms
BLINK_SPEED_SLOW_MS   = 250   # >250 ms → drowsiness / disengagement
BLINK_SPEED_WINDOW    = 10    # keep last N blink speeds

# ────────────────────────────────────────
# Gaze drift config
# ────────────────────────────────────────
GAZE_WINDOW_SEC   = 4.0       # sliding window for gaze center tracking
GAZE_CENTER_LOW   = 0.35      # iris ratio below this → off-center
GAZE_CENTER_HIGH  = 0.65      # iris ratio above this → off-center
GAZE_DRIFT_THRESH = 0.40      # if gaze_center_score < this → drifting

# ────────────────────────────────────────
# Head-pose variance config
# ────────────────────────────────────────
HEAD_WINDOW_SEC = 4.0         # sliding window length (seconds)
HEAD_STABLE_THRESH = 3.0      # std-dev below this → stable/focused (degrees)
HEAD_MOVING_THRESH = 8.0      # std-dev above this → constant movement/distraction
HEAD_TILT_THRESH   = 12.0     # |roll| above this → cognitive gap head-tilt (degrees)
NOD_MIN_DIRECTION_CHANGES = 3 # min pitch direction flips to count as nodding
NOD_PITCH_DELTA = 3.0         # min pitch change (degrees) to count a direction flip
FAKE_NOD_MIN_FLIPS = 5        # ≥5 regular-interval flips → probably faking
FAKE_NOD_REGULARITY = 0.3     # max std-dev of inter-flip intervals (seconds) for "regular"

# ────────────────────────────────────────
# Brow-down (blendshape) config
# ────────────────────────────────────────
BROW_DOWN_THRESH = 0.15       # slight brow-down > this → mental effort

def compute_ear(landmarks, eye_indices, w, h):
    """Compute Eye Aspect Ratio for one eye."""
    def _pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    p1, p2, p3, p4, p5, p6 = [_pt(i) for i in eye_indices]
    vertical_a = np.linalg.norm(p2 - p6)
    vertical_b = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.3
    return (vertical_a + vertical_b) / (2.0 * horizontal)

# ────────────────────────────────────────
# Temporal state – tracked per session
# ────────────────────────────────────────
class SessionState:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.session_start_time = time.time()  # when the session began
        self.last_emotion = None
        self.emotion_start_time = time.time()
        self.emotion_duration = 0.0
        self.surprise_timestamps: list = []
        self.state_history: list = []          # (timestamp, mapped_state) for adaptive speed

        # Blink tracking
        self.blink_timestamps: list = []     # timestamps of each detected blink
        self.ear_below_count = 0             # consecutive frames with EAR below threshold
        self.eye_closed_since = None         # when did the eye close (None = open)
        self.eye_closed_duration = 0.0       # current closure duration in seconds

        # Blink speed tracking (eyelid velocity)
        self._blink_close_time = None        # when did EAR first drop below threshold
        self.blink_speeds: deque = deque(maxlen=BLINK_SPEED_WINDOW)  # durations in ms

        # Gaze center history  (timestamp, center_score 0-1)
        self.gaze_history: deque = deque()

        # Head-pose history  (timestamp, pitch, yaw, tilt)
        self.head_history: deque = deque()
        self._last_pitch_dir = 0             # +1 rising, -1 falling
        self._nod_flips: list = []           # timestamps of pitch direction changes

    def update_emotion(self, emotion: str):
        now = time.time()
        if emotion == self.last_emotion:
            self.emotion_duration = now - self.emotion_start_time
        else:
            self.emotion_start_time = now
            self.emotion_duration = 0.0
            self.last_emotion = emotion

        if emotion == "surprise":
            self.surprise_timestamps.append(now)
        self.surprise_timestamps = [t for t in self.surprise_timestamps if now - t < 10]

    def update_blink(self, ear: float):
        """Track blinks, eye-closure duration, and blink speed."""
        now = time.time()
        if ear < EAR_THRESHOLD:
            self.ear_below_count += 1
            if self.eye_closed_since is None:
                self.eye_closed_since = now
                self._blink_close_time = now      # record when the blink started
            self.eye_closed_duration = now - self.eye_closed_since
        else:
            # Eyes just opened — was it a blink?
            if self.ear_below_count >= EAR_CONSEC_FRAMES:
                self.blink_timestamps.append(now)
                # Record blink speed (close→open duration in ms)
                if self._blink_close_time is not None:
                    speed_ms = (now - self._blink_close_time) * 1000.0
                    self.blink_speeds.append(speed_ms)
            self.ear_below_count = 0
            self.eye_closed_since = None
            self.eye_closed_duration = 0.0
            self._blink_close_time = None

        # Keep only blinks from the last 60 seconds
        self.blink_timestamps = [t for t in self.blink_timestamps if now - t < 60]

    # ── Head-pose variance ──
    def update_head_pose(self, pitch: float, yaw: float, tilt: float = 0.0):
        """Record head pose sample and detect nodding."""
        now = time.time()
        self.head_history.append((now, pitch, yaw, tilt))
        # Prune old entries outside the window
        cutoff = now - HEAD_WINDOW_SEC
        while self.head_history and self.head_history[0][0] < cutoff:
            self.head_history.popleft()

        # Nod detection: track pitch direction changes
        if len(self.head_history) >= 2:
            prev_pitch = self.head_history[-2][1]
            delta = pitch - prev_pitch
            if abs(delta) > NOD_PITCH_DELTA:
                new_dir = 1 if delta > 0 else -1
                if new_dir != self._last_pitch_dir and self._last_pitch_dir != 0:
                    self._nod_flips.append(now)
                self._last_pitch_dir = new_dir

        # Keep only recent nod flips
        self._nod_flips = [t for t in self._nod_flips if now - t < HEAD_WINDOW_SEC]

    def update_gaze_center(self, iris_ratio: float):
        """Record how centered the gaze is (0-1 ratio)."""
        now = time.time()
        # Convert iris ratio to a center score: 0.5 is perfect center, 0/1 are edges
        center_score = 1.0 - abs(iris_ratio - 0.5) * 2.0
        center_score = max(0.0, min(1.0, center_score))
        self.gaze_history.append((now, center_score))
        cutoff = now - GAZE_WINDOW_SEC
        while self.gaze_history and self.gaze_history[0][0] < cutoff:
            self.gaze_history.popleft()

    @property
    def head_pitch_std(self) -> float:
        if len(self.head_history) < 3:
            return 0.0
        pitches = [p for _, p, _, *_ in self.head_history]
        return float(np.std(pitches))

    @property
    def head_yaw_std(self) -> float:
        if len(self.head_history) < 3:
            return 0.0
        yaws = [y for _, _, y, *_ in self.head_history]
        return float(np.std(yaws))

    @property
    def is_head_stable(self) -> bool:
        return self.head_pitch_std < HEAD_STABLE_THRESH and self.head_yaw_std < HEAD_STABLE_THRESH

    @property
    def is_head_moving(self) -> bool:
        return self.head_pitch_std > HEAD_MOVING_THRESH or self.head_yaw_std > HEAD_MOVING_THRESH

    @property
    def is_nodding(self) -> bool:
        return len(self._nod_flips) >= NOD_MIN_DIRECTION_CHANGES

    @property
    def surprise_spikes(self) -> int:
        return len(self.surprise_timestamps)

    @property
    def blinks_per_minute(self) -> float:
        """Blink rate extrapolated to blinks/min from the sliding window."""
        if not self.blink_timestamps:
            return 0.0
        window = time.time() - self.blink_timestamps[0]
        if window < 1:
            return 0.0
        return len(self.blink_timestamps) / (window / 60.0)

    @property
    def is_eye_closed_long(self) -> bool:
        return self.eye_closed_duration >= LONG_CLOSURE_SEC

    @property
    def avg_blink_speed_ms(self) -> float:
        """Average blink speed in milliseconds over recent window."""
        if not self.blink_speeds:
            return 0.0
        return float(np.mean(self.blink_speeds))

    @property
    def is_blink_slow(self) -> bool:
        """True when blinks are slow (>250ms) → drowsiness / disengagement."""
        return self.avg_blink_speed_ms > BLINK_SPEED_SLOW_MS and len(self.blink_speeds) >= 2

    @property
    def gaze_center_score(self) -> float:
        """0-1 score: how centered the gaze has been over the sliding window."""
        if not self.gaze_history:
            return 1.0  # assume centered when no data
        scores = [s for _, s in self.gaze_history]
        return float(np.mean(scores))

    @property
    def is_gaze_drifting(self) -> bool:
        """True when gaze has been moving off-center over the window."""
        return self.gaze_center_score < GAZE_DRIFT_THRESH and len(self.gaze_history) >= 3

    @property
    def latest_tilt(self) -> float:
        """Latest head roll/tilt value in degrees."""
        if not self.head_history:
            return 0.0
        return self.head_history[-1][3] if len(self.head_history[-1]) > 3 else 0.0

    @property
    def is_head_tilted(self) -> bool:
        """True when head is tilted significantly (cognitive gap signal)."""
        return abs(self.latest_tilt) > HEAD_TILT_THRESH

    @property
    def is_fake_nodding(self) -> bool:
        """True when nods appear rhythmic/regular (social masking)."""
        if len(self._nod_flips) < FAKE_NOD_MIN_FLIPS:
            return False
        # Check regularity of inter-flip intervals
        intervals = [self._nod_flips[i+1] - self._nod_flips[i]
                     for i in range(len(self._nod_flips)-1)]
        if not intervals:
            return False
        return float(np.std(intervals)) < FAKE_NOD_REGULARITY

session = SessionState()

# ────────────────────────────────────────
# Head pose estimation
# ────────────────────────────────────────
MODEL_POINTS_3D = np.array([
    (0.0,   0.0,    0.0),
    (0.0,  -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# MediaPipe landmark indices for pose
POSE_LM_IDS = [1, 152, 263, 33, 287, 57]

def estimate_head_pose(landmarks, frame_w, frame_h):
    """Return (pitch, yaw, tilt) in degrees."""
    image_points = np.array([
        (landmarks[idx].x * frame_w, landmarks[idx].y * frame_h)
        for idx in POSE_LM_IDS
    ], dtype=np.float64)

    focal = frame_w
    center = (frame_w / 2, frame_h / 2)
    cam_mat = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS_3D, image_points, cam_mat, np.zeros((4,1)))
    if not ok:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = math.degrees(math.atan2(-R[2,0], sy))
    yaw   = math.degrees(math.atan2(R[2,1], R[2,2]))
    tilt  = math.degrees(math.atan2(R[1,0], R[0,0]))
    return pitch, yaw, tilt

# ────────────────────────────────────────
# Gaze estimation
# ────────────────────────────────────────
LEFT_IRIS   = [468, 469, 470, 471, 472]
RIGHT_IRIS  = [473, 474, 475, 476, 477]
LEFT_EYE_H  = [33, 133]
RIGHT_EYE_H = [362, 263]

def estimate_gaze(landmarks, pitch):
    """Return (gaze_label, avg_iris_ratio).
    gaze_label: ON_SCREEN, DOWN, AWAY, or DRIFTING.
    avg_iris_ratio: 0-1 (0.5 = perfectly centered).
    """
    def iris_ratio(iris_ids, eye_h_ids):
        iris_x = np.mean([landmarks[i].x for i in iris_ids])
        lx = landmarks[eye_h_ids[0]].x
        rx = landmarks[eye_h_ids[1]].x
        if abs(rx - lx) < 1e-6:
            return 0.5
        return (iris_x - lx) / (rx - lx)

    try:
        avg = (iris_ratio(LEFT_IRIS, LEFT_EYE_H) + iris_ratio(RIGHT_IRIS, RIGHT_EYE_H)) / 2
        if pitch > 15:
            return "DOWN", avg
        if avg < 0.25 or avg > 0.75:
            return "AWAY", avg
        return "ON_SCREEN", avg
    except:
        return "ON_SCREEN", 0.5

# ────────────────────────────────────────
# Brow-down detection from blendshapes
# ────────────────────────────────────────
def get_brow_down_score(blendshapes):
    """Extract average brow-down score from MediaPipe blendshapes.
    Returns 0.0 if blendshapes unavailable.
    """
    if not blendshapes:
        return 0.0
    scores = {}
    for bs in blendshapes:
        scores[bs.category_name] = bs.score
    left  = scores.get('browDownLeft', 0.0)
    right = scores.get('browDownRight', 0.0)
    return (left + right) / 2.0

def get_eye_squint_score(blendshapes):
    """Extract average eye-squint score from MediaPipe blendshapes."""
    if not blendshapes:
        return 0.0
    scores = {}
    for bs in blendshapes:
        scores[bs.category_name] = bs.score
    left  = scores.get('eyeSquintLeft', 0.0)
    right = scores.get('eyeSquintRight', 0.0)
    return (left + right) / 2.0

# ────────────────────────────────────────
# Core classification  (3-state cognitive model)
# ────────────────────────────────────────
# Blink rate reference (adults): ~15-20 blinks/min is normal
BLINK_LOW  = 8      # below → intense focus
BLINK_HIGH = 26     # above → fatigue / overload

# Engagement Index (EI) weights
EI_W_EXPRESSION = 0.5    # facial expression
EI_W_STABILITY  = 0.3    # head stability
EI_W_BLINK      = 0.2    # blink pattern

# Emotion-to-expression score mapping (0-1)
EXPRESSION_SCORES = {
    'happy':    0.9,
    'neutral':  0.7,
    'surprise': 0.5,
    'sad':      0.3,
    'fear':     0.2,
    'angry':    0.15,
    'disgust':  0.1,
}

def compute_engagement_index(emotion, confidence, head_stable, head_moving, blink_rate):
    """Compute Engagement Index EI ∈ [0, 1].
    EI = (0.5 × Expression) + (0.3 × Head Stability) + (0.2 × Blink Pattern)
    """
    # ── Expression score: based on emotion type × confidence ──
    base_expr = EXPRESSION_SCORES.get(emotion, 0.5)
    expression = base_expr * confidence  # scale by how confident the detection is

    # ── Head Stability score: 1.0 = perfectly stable, 0.0 = very restless ──
    if head_stable:
        stability = 1.0
    elif head_moving:
        stability = 0.0
    else:
        stability = 0.5   # in-between / measuring

    # ── Blink Pattern score: 1.0 = ideal (~15 bpm), 0.0 = extreme deviation ──
    ideal_blink = 15.0
    if blink_rate > 0:
        deviation = abs(blink_rate - ideal_blink) / ideal_blink
        blink_score = max(0.0, 1.0 - deviation)
    else:
        blink_score = 0.5  # no data yet

    ei = (EI_W_EXPRESSION * expression) + (EI_W_STABILITY * stability) + (EI_W_BLINK * blink_score)
    return max(0.0, min(1.0, ei))

def classify_understanding(emotion, conf, gaze, pitch, tilt,
                           duration, surprise_spikes,
                           blink_rate, eye_closed_long,
                           head_stable, head_moving, nodding,
                           blink_slow=False, gaze_drifting=False,
                           head_tilted=False, brow_down=0.0,
                           eye_squint=0.0, fake_nodding=False):
    """Three-state cognitive model with behavioural priority flow.

    States: Understanding (Active Acquisition),
            Confused (Cognitive Gap),
            Lost (Mental Dropout),
            Distracted, Frustrated, Unclear
    """
    # ── 0) Absolute override: prolonged eye closure → distraction ──
    if eye_closed_long:
        return "Distracted"

    blink_intense  = (0 < blink_rate < BLINK_LOW)       # very low blink → intense focus
    blink_overload = (blink_rate > BLINK_HIGH)           # very high blink → fatigue/overload
    blink_normal   = (not blink_intense and not blink_overload and blink_rate > 0)

    # ── 1) Mental Dropout → "Lost" ──
    # Gaze drifts away + slow/heavy blinks + head drooping/rotating away
    if gaze_drifting and blink_slow:
        return "Lost"
    if gaze in ["AWAY", "DOWN"] and blink_slow and head_moving:
        return "Lost"
    if gaze_drifting and head_moving:
        return "Lost"

    # ── 2) Active Acquisition → "Understanding" ──
    # Stable head + suppressed blinking (intense focus / "stare")
    if head_stable and blink_intense:
        return "Understanding"
    # Stable head + normal blink + on-screen gaze
    if head_stable and blink_normal and gaze == "ON_SCREEN":
        return "Understanding"
    # Brow-down concentration + stable head (V-shape brow = mental effort)
    if brow_down > BROW_DOWN_THRESH and head_stable:
        return "Understanding"
    # Genuine single/sharp nod (not fake rhythmic)
    if nodding and not fake_nodding:
        return "Understanding"

    # ── 3) Cognitive Gap → "Confused" ──
    # Head tilt (roll) = biological re-orientation signal
    if head_tilted and eye_squint > 0.15:
        return "Confused"
    if head_tilted:
        return "Confused"
    # Blink overload (high blink rate = cognitive strain)
    if blink_overload:
        return "Confused"
    # Surprise spikes (repeated surprise = missing key concepts)
    if surprise_spikes >= 2:
        return "Confused"
    # Sad/fear emotion + tilted head
    if emotion in ["sad", "fear"] and conf > 0.4 and abs(tilt) > 10:
        return "Confused"

    # ── 4) Distracted ──
    # Restless head (high variance movement)
    if head_moving:
        return "Distracted"
    # Fake nodding (rhythmic, repetitive = social masking)
    if fake_nodding:
        return "Distracted"
    # Neutral face + looking away
    if emotion == "neutral" and gaze in ["DOWN", "AWAY"]:
        return "Distracted"

    # ── 5) Emotion-based fallbacks ──
    if emotion == "angry" and conf > 0.5 and duration > 5:
        return "Frustrated"
    if emotion in ["neutral", "happy"] and gaze == "ON_SCREEN" and abs(pitch) < 10:
        return "Understanding"

    return "Unclear"

def blink_status_label(blink_rate, eye_closed_long, blink_slow=False):
    """Human-readable blink status string."""
    if eye_closed_long:
        return "Eyes Closed"
    if blink_rate == 0:
        return "Measuring…"
    if blink_slow:
        return "Slow Blinks (Drowsy)"
    if blink_rate < BLINK_LOW:
        return "Intense Focus"
    if blink_rate > BLINK_HIGH:
        return "Fatigue / Overload"
    return "Normal (Attentive)"

def head_pose_status_label(head_stable, head_moving, nodding,
                           head_tilted=False, fake_nodding=False):
    """Human-readable head-pose status string."""
    if fake_nodding:
        return "Fake Nodding ⚠"
    if nodding:
        return "Nodding ✓"
    if head_tilted:
        return "Head Tilt 🤔"
    if head_moving:
        return "Restless"
    if head_stable:
        return "Stable (Focused)"
    return "Measuring…"

# ────────────────────────────────────────
# 3-state mapping  (backend mirror of frontend logic)
# ────────────────────────────────────────
def map_to_three_states(state: str) -> str:
    """Map any classification to the 3 cognitive states."""
    mapping = {
        'Understanding': 'Understanding',
        'Confused':      'Confused',
        'Frustrated':    'Confused',
        'Lost':          'Lost',
        'Distracted':    'Lost',
        'Unclear':       'Confused',
    }
    return mapping.get(state, state)

# Speed values for the 3 cognitive states
THREE_STATE_SPEED = {
    'Understanding': 1.3,
    'Confused':      0.7,
    'Lost':          0.5,
}

# ────────────────────────────────────────
# Adaptive speed control
# ────────────────────────────────────────
SPEED_INIT_WINDOW  = 10.0    # initial observation window (seconds)
SPEED_ROLL_WINDOW  = 10.0    # rolling window for ongoing speed updates (seconds)

def get_adaptive_speed(state_history, session_start, current_state):
    """Compute speed based on dominant state in the time window.

    - First 10 seconds: accumulate states, use dominant state speed.
    - After 10 seconds: rolling 10-second window dominant state speed.
    """
    now = time.time()
    elapsed = now - session_start

    if elapsed < SPEED_INIT_WINDOW:
        # Still in initial window — use all history so far
        if not state_history:
            return 1.0  # default until we have data
        # Find the dominant state
        counts: dict = {}
        for ts, st in state_history:
            counts[st] = counts.get(st, 0) + 1
        dominant = max(counts, key=counts.get)
        return THREE_STATE_SPEED.get(dominant, 1.0)
    else:
        # Rolling window — only consider last N seconds
        cutoff = now - SPEED_ROLL_WINDOW
        recent = [(ts, st) for ts, st in state_history if ts >= cutoff]
        if not recent:
            return THREE_STATE_SPEED.get(current_state, 1.0)
        counts: dict = {}
        for ts, st in recent:
            counts[st] = counts.get(st, 0) + 1
        dominant = max(counts, key=counts.get)
        return THREE_STATE_SPEED.get(dominant, 1.0)

# ────────────────────────────────────────
# Main analysis pipeline
# ────────────────────────────────────────
def analyze_frame_full(frame):
    """Run emotion + gaze + pose + blink + blendshapes + head-variance → classification."""
    result = {
        'face_detected': False,
        'emotion': 'neutral', 'confidence': 0.0, 'all_emotions': {},
        'gaze': 'ON_SCREEN', 'pitch': 0.0, 'yaw': 0.0, 'tilt': 0.0,
        'duration': 0.0, 'surprise_spikes': 0,
        'blink_rate': 0.0, 'eye_closed': False, 'blink_status': '—',
        'blink_speed': 0.0, 'blink_slow': False,
        'head_pitch_std': 0.0, 'head_yaw_std': 0.0,
        'head_stable': False, 'head_moving': False, 'nodding': False,
        'head_tilted': False, 'fake_nodding': False,
        'head_pose_status': '—',
        'brow_down': 0.0, 'eye_squint': 0.0,
        'gaze_center_score': 1.0, 'gaze_drifting': False,
        'engagement_index': 0.0,
        'understanding': 'No One Available', 'speed': 1.0,
    }

    # ── Check for a face via MediaPipe first ──
    face_landmarks = None
    face_blendshapes = None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = face_landmarker.detect(mp_image)
        if detection.face_landmarks:
            face_landmarks = detection.face_landmarks[0]
        if detection.face_blendshapes:
            face_blendshapes = detection.face_blendshapes[0]
    except Exception as e:
        print(f"MediaPipe error: {e}")

    # No face? Return early with "No One Available"
    if face_landmarks is None:
        return result

    result['face_detected'] = True
    h, w, _ = frame.shape

    # 1) Emotion via DeepFace
    try:
        df = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
        if isinstance(df, list):
            df = df[0] if df else {}
        edata = df.get('emotion', {})
        dom = df.get('dominant_emotion', 'neutral')
        conf = edata.get(dom, 0.0) / 100.0
        result['emotion'] = dom
        result['confidence'] = float(conf)
        result['all_emotions'] = {k: float(v)/100.0 for k, v in edata.items()}
    except Exception as e:
        print(f"DeepFace error: {e}")

    # 2) Gaze + head pose + blink
    pitch, yaw, tilt = estimate_head_pose(face_landmarks, w, h)
    gaze, iris_ratio = estimate_gaze(face_landmarks, pitch)

    ear_left  = compute_ear(face_landmarks, LEFT_EYE_IDX, w, h)
    ear_right = compute_ear(face_landmarks, RIGHT_EYE_IDX, w, h)
    ear = (ear_left + ear_right) / 2.0

    result['pitch'] = round(pitch, 1)
    result['yaw'] = round(yaw, 1)
    result['tilt'] = round(tilt, 1)
    result['gaze'] = gaze

    # 3) Blink tracking (rate + speed)
    session.update_blink(ear)
    blink_rate = session.blinks_per_minute
    eye_closed_long = session.is_eye_closed_long
    blink_slow = session.is_blink_slow
    result['blink_rate'] = round(blink_rate, 1)
    result['eye_closed'] = eye_closed_long
    result['blink_speed'] = round(session.avg_blink_speed_ms, 0)
    result['blink_slow'] = blink_slow
    result['blink_status'] = blink_status_label(blink_rate, eye_closed_long, blink_slow)

    # 4) Gaze drift tracking
    session.update_gaze_center(iris_ratio)
    gaze_drifting = session.is_gaze_drifting
    result['gaze_center_score'] = round(session.gaze_center_score, 2)
    result['gaze_drifting'] = gaze_drifting

    # 5) Head-pose variance tracking (now includes tilt)
    session.update_head_pose(pitch, yaw, tilt)
    head_stable = session.is_head_stable
    head_moving = session.is_head_moving
    nodding = session.is_nodding
    head_tilted = session.is_head_tilted
    fake_nodding = session.is_fake_nodding
    result['head_pitch_std'] = round(session.head_pitch_std, 2)
    result['head_yaw_std'] = round(session.head_yaw_std, 2)
    result['head_stable'] = head_stable
    result['head_moving'] = head_moving
    result['nodding'] = nodding
    result['head_tilted'] = head_tilted
    result['fake_nodding'] = fake_nodding
    result['head_pose_status'] = head_pose_status_label(
        head_stable, head_moving, nodding, head_tilted, fake_nodding)

    # 6) Blendshape signals (brow-down + eye squint)
    brow_down = get_brow_down_score(face_blendshapes)
    eye_squint = get_eye_squint_score(face_blendshapes)
    result['brow_down'] = round(brow_down, 3)
    result['eye_squint'] = round(eye_squint, 3)

    # 7) Temporal tracking
    session.update_emotion(result['emotion'])
    result['duration'] = round(session.emotion_duration, 1)
    result['surprise_spikes'] = session.surprise_spikes

    # 8) Engagement Index (EI)
    ei = compute_engagement_index(
        result['emotion'], result['confidence'],
        head_stable, head_moving, blink_rate)
    result['engagement_index'] = round(ei, 2)

    # 9) Classification (3-state cognitive model + fallbacks)
    understanding = classify_understanding(
        result['emotion'], result['confidence'],
        gaze, pitch, tilt, session.emotion_duration, session.surprise_spikes,
        blink_rate, eye_closed_long,
        head_stable, head_moving, nodding,
        blink_slow=blink_slow,
        gaze_drifting=gaze_drifting,
        head_tilted=head_tilted,
        brow_down=brow_down,
        eye_squint=eye_squint,
        fake_nodding=fake_nodding,
    )
    result['understanding'] = understanding

    # 10) Adaptive speed (based on dominant state in rolling window)
    mapped_state = map_to_three_states(understanding)
    session.state_history.append((time.time(), mapped_state))
    # Prune old state history (keep last 60 seconds max)
    cutoff = time.time() - 60.0
    session.state_history = [(t, s) for t, s in session.state_history if t >= cutoff]

    speed = get_adaptive_speed(session.state_history, session.session_start_time, mapped_state)
    result['speed'] = round(speed, 2)
    
    # Calculate dominant state for frontend (smoothed/confirmed state)
    # Use last 10 seconds (STATE_WINDOW_SEC from frontend)
    confirm_window = 10.0
    confirm_cutoff = time.time() - confirm_window
    recent_states = [s for t, s in session.state_history if t >= confirm_cutoff]
    
    if recent_states:
        from collections import Counter
        counts = Counter(recent_states)
        confirmed_state = counts.most_common(1)[0][0]
    else:
        confirmed_state = mapped_state
        
    result['confirmed_state'] = confirmed_state
    result['dominant_state'] = mapped_state # Raw mapped state for this frame

    # Log to database
    try:
        log_frame(result, session.session_id)
    except Exception as e:
        print(f"Database logging error: {e}")

    return result

# ────────────────────────────────────────
# Routes
# ────────────────────────────────────────
@app.route('/api/analyze-frame', methods=['POST'])
def api_analyze_frame():
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame provided'}), 400
        image_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return jsonify(analyze_frame_full(frame))
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/stream', methods=['GET'])
def stream_video():
    try:
        if not os.path.exists(VIDEO_PATH):
            return jsonify({'error': 'Video file not found'}), 404
        file_size = os.path.getsize(VIDEO_PATH)
        range_header = request.headers.get('Range', None)
        if range_header:
            parts = range_header.replace('bytes=', '').split('-')
            byte_start = int(parts[0])
            byte_end = int(parts[1]) if parts[1] else file_size - 1
            length = byte_end - byte_start + 1
            with open(VIDEO_PATH, 'rb') as f:
                f.seek(byte_start)
                data = f.read(length)
            resp = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
            resp.headers.add('Content-Range', f'bytes {byte_start}-{byte_end}/{file_size}')
            resp.headers.add('Accept-Ranges', 'bytes')
            resp.headers.add('Content-Length', str(length))
            return resp
        return send_file(VIDEO_PATH, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'message': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('analyze_frame')
def handle_frame(data):
    try:
        image_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        emit('emotion_result', analyze_frame_full(frame))
    except Exception as e:
        print(f"WebSocket error: {e}")
        emit('error', {'error': str(e)})

@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    """Return live session stats for the dashboard."""
    try:
        # Build stats from in-memory session state
        state_counts = {}
        for ts, st in session.state_history:
            state_counts[st] = state_counts.get(st, 0) + 1
        total = sum(state_counts.values())

        # Current session elapsed time
        elapsed = time.time() - session.session_start_time

        return jsonify({
            'session_id': session.session_id,
            'elapsed_seconds': round(elapsed, 1),
            'total_frames': total,
            'state_counts': state_counts,
            'current_state': session.state_history[-1][1] if session.state_history else 'No One Available',
            'blink_rate': round(session.blinks_per_minute, 1),
            'engagement_index': round(
                compute_engagement_index(
                    session.last_emotion or 'neutral', 0.5,
                    session.is_head_stable, session.is_head_moving,
                    session.blinks_per_minute
                ), 2
            ),
            'head_stable': session.is_head_stable,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/hourly-reports', methods=['GET'])
def dashboard_hourly_reports():
    """Fetch hourly reports from the last 24 hours."""
    try:
        conn = get_db_connection(DB_NAME)
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM hourly_reports
            WHERE report_end_time >= NOW() - INTERVAL 24 HOUR
            ORDER BY report_end_time DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        # Convert datetime objects to strings
        for row in rows:
            for key in ['report_start_time', 'report_end_time']:
                if row.get(key):
                    row[key] = row[key].isoformat()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/sessions', methods=['GET'])
def dashboard_sessions():
    """Fetch distinct sessions with summary stats."""
    try:
        conn = get_db_connection(DB_NAME)
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                session_id,
                COUNT(*) as total_frames,
                AVG(engagement_index) as avg_engagement,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                (
                    SELECT understanding_state FROM frame_logs f2
                    WHERE f2.session_id = f1.session_id
                    GROUP BY understanding_state
                    ORDER BY COUNT(*) DESC
                    LIMIT 1
                ) as dominant_state
            FROM frame_logs f1
            GROUP BY session_id
            ORDER BY MAX(timestamp) DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        for row in rows:
            for key in ['start_time', 'end_time']:
                if row.get(key):
                    row[key] = row[key].isoformat()
            if row.get('avg_engagement') is not None:
                row['avg_engagement'] = round(float(row['avg_engagement']), 2)
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'video_exists': os.path.exists(VIDEO_PATH)})

if __name__ == '__main__':
    print(f"Video path: {VIDEO_PATH}")
    print(f"Video exists: {os.path.exists(VIDEO_PATH)}")
    init_db()
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
