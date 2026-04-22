"""
Hand tracker for EMS stimulation experiments.

Runs a camera (phone via Continuity Camera) in a background thread and provides:
- Finger joint angle tracking (per-finger flexion at MCP, PIP, DIP)
- Automatic baseline capture before stimulation
- Automatic peak measurement after stimulation

This is the FINGER-ONLY tracker. Wrist angle is handled by PoseTracker.

Usage in main_window.py:
    from core.vision.hand_tracker import HandTracker

    # On session start
    self.hand_tracker = HandTracker(camera_index=1)  # phone camera
    self.hand_tracker.start()

    # Before stimulation:
    self.hand_tracker.capture_baseline()
    # ... EMS fires ...
    result = self.hand_tracker.measure_movement()

    # On session stop
    self.hand_tracker.stop()
"""

import cv2
import numpy as np
import threading
import time
import os
import urllib.request
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque

from core.vision.undistorter import Undistorter, CalibrationNotFoundError

# MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Run: pip install mediapipe")


# =========================================================================
# Hand landmark definitions
# =========================================================================

class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

FINGER_JOINTS = {
    "thumb": [
        (HandLandmark.WRIST, HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
        (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
        (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    ],
    "index": [
        (HandLandmark.WRIST, HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP),
        (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP),
        (HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP),
    ],
    "middle": [
        (HandLandmark.WRIST, HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP),
        (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP),
        (HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP),
    ],
    "ring": [
        (HandLandmark.WRIST, HandLandmark.RING_MCP, HandLandmark.RING_PIP),
        (HandLandmark.RING_MCP, HandLandmark.RING_PIP, HandLandmark.RING_DIP),
        (HandLandmark.RING_PIP, HandLandmark.RING_DIP, HandLandmark.RING_TIP),
    ],
    "pinky": [
        (HandLandmark.WRIST, HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
        (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
        (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
    ],
}

JOINT_NAMES = {
    "thumb": ["CMC", "MCP", "IP"],
    "index": ["MCP", "PIP", "DIP"],
    "middle": ["MCP", "PIP", "DIP"],
    "ring": ["MCP", "PIP", "DIP"],
    "pinky": ["MCP", "PIP", "DIP"],
}

Landmarks = List[Tuple[float, float, float]]


# =========================================================================
# Angle calculations
# =========================================================================

def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle at p2 formed by p1-p2-p3 (degrees)."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _calculate_finger_angles(landmarks: Landmarks) -> Dict[str, List[float]]:
    """Calculate all joint angles for each finger."""
    pts = [np.array(lm) for lm in landmarks]
    result = {}
    for finger in FINGER_NAMES:
        angles = []
        for i1, i2, i3 in FINGER_JOINTS[finger]:
            angles.append(_calculate_angle(pts[i1], pts[i2], pts[i3]))
        result[finger] = angles
    return result


def _calculate_total_flexion(finger_angles: Dict[str, List[float]]) -> Dict[str, float]:
    """Sum of joint angles per finger."""
    return {f: sum(a) for f, a in finger_angles.items()}


# =========================================================================
# Result data class
# =========================================================================

@dataclass
class HandMovementResult:
    """Movement result for finger tracking from one stimulation event."""

    timestamp: Optional[datetime] = None
    latency_ms: float = 0.0

    # Finger angles (baseline vs result)
    baseline_finger_angles: Dict[str, List[float]] = field(default_factory=dict)
    result_finger_angles: Dict[str, List[float]] = field(default_factory=dict)
    finger_angle_deltas: Dict[str, List[float]] = field(default_factory=dict)

    # Finger flexion summary
    baseline_flexion: Dict[str, float] = field(default_factory=dict)
    result_flexion: Dict[str, float] = field(default_factory=dict)
    flexion_change: Dict[str, float] = field(default_factory=dict)

    # Overall
    total_finger_movement: float = 0.0
    primary_finger: str = ""
    primary_finger_movement: float = 0.0
    movement_type: str = ""  # "flexion", "extension", "mixed", "none"

    # Quality
    hand_detected: bool = False
    confidence: float = 0.0

    # Stimulation params (filled by caller)
    channel: Optional[int] = None
    intensity_ma: Optional[int] = None
    pulse_width_us: Optional[int] = None

    def to_dict(self) -> dict:
        """Export for database storage."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latency_ms": self.latency_ms,
            "finger_angle_deltas": self.finger_angle_deltas,
            "flexion_change": self.flexion_change,
            "total_finger_movement": self.total_finger_movement,
            "primary_finger": self.primary_finger,
            "primary_finger_movement": self.primary_finger_movement,
            "movement_type": self.movement_type,
            "hand_detected": self.hand_detected,
            "confidence": self.confidence,
            "channel": self.channel,
            "intensity_ma": self.intensity_ma,
            "pulse_width_us": self.pulse_width_us,
        }


# =========================================================================
# Hand Tracker class
# =========================================================================

class HandTracker:
    """
    Webcam-based finger movement tracker for EMS experiments.

    Uses MediaPipe Hand Landmarker on a phone camera (zoomed on fingers).
    Tracks per-finger joint angles (MCP, PIP, DIP) and total flexion.

    Does NOT track wrist angle — that is handled by PoseTracker.
    """

    # Hand skeleton connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    def __init__(
        self,
        camera_index: int = 1,
        smoothing_window: int = 5,
        model_path: Optional[str] = None,
        show_preview: bool = True,
        save_dir: str = "captures/hand",
    ):
        """
        Args:
            camera_index: Camera index (1 = phone via Continuity Camera typically).
            smoothing_window: Frames to average for noise reduction.
            model_path: Path to hand_landmarker.task (auto-downloads if None).
            show_preview: Store annotated preview frames for GUI display.
            save_dir: Directory to save baseline/result frame images.
        """
        self.camera_index = camera_index
        self.smoothing_window = smoothing_window
        self.show_preview = show_preview
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stim_count = 0

        # Latest frame data (updated by background thread)
        self._latest_landmarks: Optional[Landmarks] = None
        self._latest_confidence: float = 0.0
        self._hand_detected: bool = False
        self._preview_frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None

        # Smoothing buffer
        self._landmarks_buffer: deque = deque(maxlen=smoothing_window)

        # Baseline
        self._baseline_landmarks: Optional[Landmarks] = None
        self._baseline_finger_angles: Dict[str, List[float]] = {}
        self._baseline_flexion: Dict[str, float] = {}
        self._baseline_timestamp: Optional[datetime] = None
        self._baseline_set: bool = False

        # Last result (for preview overlay)
        self._last_result: Optional[HandMovementResult] = None

        # MediaPipe
        self._model_path = model_path
        self._detector = None
        self._undistorter: Optional[Undistorter] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the camera and hand detection thread."""
        if not MEDIAPIPE_AVAILABLE:
            print("✗ Cannot start hand tracker — MediaPipe not installed")
            return False

        # Load calibration — raises CalibrationNotFoundError if missing
        self._undistorter = Undistorter(self.camera_index)

        self._model_path = self._ensure_model(self._model_path)
        if self._model_path is None:
            print("✗ Cannot start hand tracker — model not available")
            return False

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self._model_path),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        print(f"✓ Hand tracker started (camera {self.camera_index} — phone/finger cam)")
        return True

    def stop(self) -> None:
        """Stop the camera and detection thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._detector:
            self._detector.close()
            self._detector = None

        self._baseline_set = False
        self._last_result = None
        print("✓ Hand tracker stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def hand_detected(self) -> bool:
        with self._lock:
            return self._hand_detected

    # ------------------------------------------------------------------
    # Baseline & Measurement
    # ------------------------------------------------------------------

    def capture_baseline(self) -> bool:
        """
        Lock the current hand pose as baseline.
        Call right before EMS stimulation.

        Returns:
            True if baseline captured (hand was detected).
        """
        with self._lock:
            if not self._hand_detected or self._latest_landmarks is None:
                print("✗ [Hand] Cannot capture baseline — no hand detected")
                return False

            if len(self._landmarks_buffer) >= 3:
                avg_landmarks = self._average_landmarks(list(self._landmarks_buffer))
            else:
                avg_landmarks = self._latest_landmarks

            self._baseline_landmarks = avg_landmarks
            self._baseline_timestamp = datetime.now()
            self._baseline_finger_angles = _calculate_finger_angles(avg_landmarks)
            self._baseline_flexion = _calculate_total_flexion(self._baseline_finger_angles)
            self._baseline_set = True
            self._last_result = None

        self._stim_count += 1

        # Save baseline frame
        path = self._save_frame("baseline", self._baseline_landmarks,
                                extra_text="BASELINE — before stimulation")
        if path:
            print(f"  📸 [Hand] Baseline frame saved: {path}")

        print(f"✓ [Hand] Baseline captured at "
              f"{self._baseline_timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        return True

    def measure_movement(
        self,
        recording_duration: float = 2.0,
        channel: Optional[int] = None,
        intensity: Optional[int] = None,
        pulse_width: Optional[int] = None,
    ) -> Optional[HandMovementResult]:
        """
        Record frames and find PEAK finger movement from baseline.

        Args:
            recording_duration: How long to record in seconds.
            channel: EMS channel (for logging).
            intensity: EMS intensity in mA (for logging).
            pulse_width: EMS pulse width in μs (for logging).

        Returns:
            HandMovementResult from the peak frame, or None.
        """
        if self._baseline_landmarks is None:
            print("✗ [Hand] Cannot measure — no baseline set")
            return None

        print(f"  [Hand] Recording for {recording_duration}s...")

        # Collect frames
        recorded_frames: List[dict] = []
        frame_snapshots: dict = {}  # index -> raw frame for saving
        start_time = time.time()
        sample_count = 0

        while (time.time() - start_time) < recording_duration:
            with self._lock:
                if not self._hand_detected or self._latest_landmarks is None:
                    time.sleep(0.01)
                    continue
                landmarks = list(self._latest_landmarks)
                confidence = self._latest_confidence
                # Save raw frame every 5th sample for peak image
                if sample_count % 5 == 0 and self._raw_frame is not None:
                    frame_snapshots[len(recorded_frames)] = self._raw_frame.copy()

            elapsed_ms = (time.time() - start_time) * 1000

            finger_angles = _calculate_finger_angles(landmarks)
            flexion = _calculate_total_flexion(finger_angles)

            flexion_change = {
                f: flexion.get(f, 0) - self._baseline_flexion.get(f, 0)
                for f in FINGER_NAMES
            }
            total_finger_dev = sum(abs(v) for v in flexion_change.values())

            recorded_frames.append({
                "landmarks": landmarks,
                "confidence": confidence,
                "elapsed_ms": elapsed_ms,
                "finger_angles": finger_angles,
                "flexion": flexion,
                "flexion_change": flexion_change,
                "total_finger_dev": total_finger_dev,
            })

            sample_count += 1
            time.sleep(0.02)  # ~50 FPS sampling

        if not recorded_frames:
            print("✗ [Hand] No frames captured during recording window")
            return None

        print(f"  ✓ [Hand] Recorded {len(recorded_frames)} frames")

        # Find peak region: average frames within 80% of max displacement
        max_dev = max(f["total_finger_dev"] for f in recorded_frames)
        threshold = max_dev * 0.8

        near_peak_frames = [f for f in recorded_frames if f["total_finger_dev"] >= threshold]

        print(f"  [Hand] Max deviation: {max_dev:.1f} deg, "
              f"threshold (80%): {threshold:.1f} deg, "
              f"near-peak frames: {len(near_peak_frames)}/{len(recorded_frames)}")

        # Average the near-peak finger angles and flexion
        avg_finger_angles = {}
        avg_flexion = {}
        avg_flexion_change = {}
        for finger in FINGER_NAMES:
            # Average per-joint angles
            joint_count = len(near_peak_frames[0]["finger_angles"].get(finger, [0, 0, 0]))
            avg_joints = []
            for j in range(joint_count):
                vals = [f["finger_angles"].get(finger, [0]*joint_count)[j] for f in near_peak_frames]
                avg_joints.append(float(np.mean(vals)))
            avg_finger_angles[finger] = avg_joints

            # Average total flexion
            vals = [f["flexion"].get(finger, 0) for f in near_peak_frames]
            avg_flexion[finger] = float(np.mean(vals))

            # Average flexion change
            vals = [f["flexion_change"].get(finger, 0) for f in near_peak_frames]
            avg_flexion_change[finger] = float(np.mean(vals))

        avg_elapsed_ms = float(np.mean([f["elapsed_ms"] for f in near_peak_frames]))
        avg_confidence = float(np.mean([f["confidence"] for f in near_peak_frames]))

        # Use the single highest frame for the saved image
        peak_frame = max(near_peak_frames, key=lambda f: f["total_finger_dev"])
        peak_idx = recorded_frames.index(peak_frame)

        print(f"  [Hand] Averaged {len(near_peak_frames)} frames near peak "
              f"({avg_elapsed_ms:.0f}ms avg latency)")

        # Build result using averaged values
        result_finger_angles = avg_finger_angles
        result_flexion = avg_flexion
        flexion_change = avg_flexion_change

        # Finger angle deltas per joint
        angle_deltas = {}
        for finger in FINGER_NAMES:
            b = self._baseline_finger_angles.get(finger, [0, 0, 0])
            r = result_finger_angles.get(finger, [0, 0, 0])
            angle_deltas[finger] = [rv - bv for bv, rv in zip(b, r)]

        total_movement = sum(sum(abs(d) for d in deltas) for deltas in angle_deltas.values())

        # Primary finger
        primary_finger = max(flexion_change, key=lambda f: abs(flexion_change[f]))
        primary_movement = abs(flexion_change[primary_finger])

        # Classify movement type
        threshold = 10.0
        pos = sum(1 for v in flexion_change.values() if v > threshold)
        neg = sum(1 for v in flexion_change.values() if v < -threshold)
        if pos == 0 and neg == 0:
            movement_type = "none"
        elif pos > 0 and neg == 0:
            movement_type = "extension"
        elif neg > 0 and pos == 0:
            movement_type = "flexion"
        else:
            movement_type = "mixed"

        result = HandMovementResult(
            timestamp=datetime.now(),
            latency_ms=avg_elapsed_ms,
            baseline_finger_angles=self._baseline_finger_angles,
            result_finger_angles=result_finger_angles,
            finger_angle_deltas=angle_deltas,
            baseline_flexion=self._baseline_flexion,
            result_flexion=result_flexion,
            flexion_change=flexion_change,
            total_finger_movement=round(total_movement, 1),
            primary_finger=primary_finger,
            primary_finger_movement=round(primary_movement, 1),
            movement_type=movement_type,
            hand_detected=True,
            confidence=avg_confidence,
            channel=channel,
            intensity_ma=intensity,
            pulse_width_us=pulse_width,
        )

        self._print_result(result)
        self._last_result = result

        result_text = (
            f"PEAK at {result.latency_ms:.0f}ms\n"
            f"Primary: {result.primary_finger} ({result.primary_finger_movement:.1f}°)\n"
            f"Total finger movement: {result.total_finger_movement:.1f}°\n"
            f"Type: {result.movement_type}"
        )

        # Save peak frame from the closest stored snapshot
        if frame_snapshots:
            closest_idx = min(frame_snapshots.keys(), key=lambda i: abs(i - peak_idx))
            peak_raw_frame = frame_snapshots[closest_idx]
            path = self._save_frame_from("peak", peak_raw_frame,
                                          peak_frame["landmarks"],
                                          extra_text=result_text)
            if path:
                print(f"  [Hand] Peak frame saved: {path}")
        else:
            path = self._save_frame("peak", peak_frame["landmarks"], extra_text=result_text)
            if path:
                print(f"  [Hand] Peak frame saved: {path} (fallback)")

        return result

    # ------------------------------------------------------------------
    # Background camera thread
    # ------------------------------------------------------------------

    def _camera_loop(self) -> None:
        """Background thread: reads camera and runs hand detection."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"✗ [Hand] Could not open camera {self.camera_index}")
            self._running = False
            return

        print(f"  [Hand] Camera {self.camera_index} opened")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = self._undistorter.apply(frame)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)

            landmarks_px = None

            with self._lock:
                self._raw_frame = frame.copy()

                if result.hand_landmarks:
                    hand_lm = result.hand_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lm]
                    landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

                    self._latest_landmarks = landmarks
                    self._hand_detected = True
                    self._landmarks_buffer.append(landmarks)

                    if result.handedness:
                        self._latest_confidence = result.handedness[0][0].score
                else:
                    self._hand_detected = False

            # Draw preview
            if self.show_preview and landmarks_px:
                self._draw_preview(frame, landmarks_px, result)

            with self._lock:
                self._preview_frame = frame.copy() if self.show_preview else None

        cap.release()
        print("  [Hand] Camera released")

    # ------------------------------------------------------------------
    # Preview & saving
    # ------------------------------------------------------------------

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame for GUI display."""
        with self._lock:
            if self._preview_frame is not None:
                return self._preview_frame.copy()
        return None

    def _draw_preview(self, frame, landmarks_px, detection_result) -> None:
        """Draw hand skeleton and status on frame."""
        h, w = frame.shape[:2]

        # Draw skeleton
        if landmarks_px:
            for s, e in self.HAND_CONNECTIONS:
                cv2.line(frame, landmarks_px[s], landmarks_px[e], (255, 255, 255), 2)

            for i, (x, y) in enumerate(landmarks_px):
                if i == 0:
                    color = (0, 0, 255)       # Wrist = red
                elif i in [4, 8, 12, 16, 20]:
                    color = (0, 255, 0)       # Tips = green
                else:
                    color = (255, 0, 0)       # Joints = blue
                cv2.circle(frame, (x, y), 5, color, -1)

            if detection_result.handedness:
                label = detection_result.handedness[0][0].category_name
                conf = detection_result.handedness[0][0].score
                cv2.putText(frame, f"{label} ({conf:.0%})", (w - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Status
        status_color = (0, 255, 0) if self._hand_detected else (0, 0, 255)
        status_text = "HAND DETECTED" if self._hand_detected else "NO HAND"
        cv2.putText(frame, f"[FINGERS] {status_text}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if self._baseline_set:
            cv2.putText(frame, "BASELINE SET", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # Last result overlay
        if self._last_result and self._last_result.hand_detected:
            r = self._last_result
            y0 = h - 100

            cv2.rectangle(frame, (10, y0 - 10), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, y0 - 10), (w - 10, h - 10), (255, 255, 255), 1)

            cv2.putText(frame, f"Fingers: {r.primary_finger} {r.primary_finger_movement:.1f}°  |  "
                        f"Total: {r.total_finger_movement:.1f}°  |  Type: {r.movement_type}",
                        (20, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            x_offset = 20
            y_line = y0 + 50
            for finger in FINGER_NAMES:
                change = r.flexion_change.get(finger, 0)
                color = (100, 255, 100) if abs(change) > 10 else (150, 150, 150)
                text = f"{finger[:3]}:{change:+.0f}°"
                cv2.putText(frame, text, (x_offset, y_line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                x_offset += 100

    def _save_frame_from(self, label: str, frame: np.ndarray,
                         landmarks: Optional[Landmarks] = None,
                         extra_text: Optional[str] = None) -> Optional[str]:
        """Save a specific raw frame with landmarks (for peak snapshots)."""
        frame = frame.copy()
        h, w = frame.shape[:2]

        if landmarks is not None:
            landmarks_px = [(int(x * w), int(y * h)) for x, y, z in landmarks]
            for s, e in self.HAND_CONNECTIONS:
                cv2.line(frame, landmarks_px[s], landmarks_px[e], (255, 255, 255), 2)
            for i, (px, py) in enumerate(landmarks_px):
                if i == 0:
                    color = (0, 0, 255)
                elif i in [4, 8, 12, 16, 20]:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.circle(frame, (px, py), 6, color, -1)
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)

        cv2.putText(frame, f"[HAND] {label.upper()}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(frame, timestamp_str, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if extra_text:
            y_pos = h - 20
            for line in reversed(extra_text.strip().split("\n")):
                cv2.putText(frame, line, (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos -= 22

        filename = f"hand_stim{self._stim_count:03d}_{label}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def _save_frame(self, label: str, landmarks: Optional[Landmarks] = None,
                    extra_text: Optional[str] = None) -> Optional[str]:
        """Save the current raw frame with landmarks drawn on it."""
        with self._lock:
            if self._raw_frame is None:
                return None
            frame = self._raw_frame.copy()

        h, w = frame.shape[:2]

        if landmarks is not None:
            landmarks_px = [(int(x * w), int(y * h)) for x, y, z in landmarks]
            for s, e in self.HAND_CONNECTIONS:
                cv2.line(frame, landmarks_px[s], landmarks_px[e], (255, 255, 255), 2)
            for i, (px, py) in enumerate(landmarks_px):
                if i == 0:
                    color = (0, 0, 255)
                elif i in [4, 8, 12, 16, 20]:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.circle(frame, (px, py), 6, color, -1)
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)

        cv2.putText(frame, f"[HAND] {label.upper()}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(frame, timestamp_str, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if extra_text:
            y_pos = h - 20
            for line in reversed(extra_text.strip().split("\n")):
                cv2.putText(frame, line, (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos -= 22

        filename = f"hand_stim{self._stim_count:03d}_{label}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _average_landmarks(self, landmarks_list: List[Landmarks]) -> Landmarks:
        """Average multiple frames of landmarks for stability."""
        arr = np.array(landmarks_list)
        avg = np.mean(arr, axis=0)
        return [tuple(row) for row in avg]

    def _ensure_model(self, path: Optional[str]) -> Optional[str]:
        """Download hand landmarker model if needed."""
        if path and os.path.exists(path):
            return path

        default_path = "hand_landmarker.task"
        if os.path.exists(default_path):
            return default_path

        print("  Downloading hand landmarker model...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        try:
            urllib.request.urlretrieve(url, default_path)
            print(f"  ✓ Model downloaded: {default_path}")
            return default_path
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return None

    def _print_result(self, r: HandMovementResult) -> None:
        """Print finger movement report to terminal."""
        print()
        print("=" * 60)
        print("  FINGER MOVEMENT RESULT (Hand Tracker)")
        print("=" * 60)

        if r.channel is not None:
            print(f"  Stimulation: Ch{r.channel}, {r.intensity_ma}mA, {r.pulse_width_us}μs")

        print(f"  Latency: {r.latency_ms:.0f} ms")
        print(f"  Confidence: {r.confidence:.0%}")
        print()

        print("  --- Finger Joint Angles (Δ from baseline) ---")
        print(f"  {'Finger':8s}  {'':>7s}  {'':>7s}  {'':>7s}  {'Total':>8s}")
        print(f"  {'-'*50}")

        for finger in FINGER_NAMES:
            deltas = r.finger_angle_deltas.get(finger, [0, 0, 0])
            total = r.flexion_change.get(finger, 0)
            jnames = JOINT_NAMES[finger]
            d_strs = [f"{jnames[i]}={d:+.1f}°" for i, d in enumerate(deltas)]
            print(f"  {finger.capitalize():8s}  {', '.join(d_strs):>35s}  {total:+7.1f}°")

        print()
        print(f"  Total finger movement: {r.total_finger_movement:.1f}°")
        print(f"  Primary finger: {r.primary_finger} ({r.primary_finger_movement:.1f}°)")
        print(f"  Movement type: {r.movement_type}")
        print("=" * 60)
        print()