"""
Movement tracker for EMS stimulation experiments.

Runs a webcam in a background thread and provides:
- Wrist angle tracking (flexion/extension)
- Finger joint angle tracking (per-finger flexion)
- Automatic baseline capture before stimulation
- Automatic measurement after stimulation

Designed to integrate with the EMS GUI via simple method calls.
All output goes to terminal (database integration later).

Usage in main_window.py:
    from core.vision.movement_tracker import MovementTracker

    # On session start
    self.movement_tracker = MovementTracker()
    self.movement_tracker.start()

    # On stimulate (called automatically):
    self.movement_tracker.capture_baseline()
    # ... EMS fires ...
    result = self.movement_tracker.measure_movement()
    # prints full report to terminal

    # On session stop
    self.movement_tracker.stop()
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
# Inline landmark definitions (standalone — no package dependency)
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

FINGERTIPS = {
    "thumb": HandLandmark.THUMB_TIP,
    "index": HandLandmark.INDEX_TIP,
    "middle": HandLandmark.MIDDLE_TIP,
    "ring": HandLandmark.RING_TIP,
    "pinky": HandLandmark.PINKY_TIP,
}

MCP_INDICES = [
    HandLandmark.INDEX_MCP,
    HandLandmark.MIDDLE_MCP,
    HandLandmark.RING_MCP,
    HandLandmark.PINKY_MCP,
]

TIP_INDICES = [
    HandLandmark.INDEX_TIP,
    HandLandmark.MIDDLE_TIP,
    HandLandmark.RING_TIP,
    HandLandmark.PINKY_TIP,
]

Landmarks = List[Tuple[float, float, float]]


# =========================================================================
# Angle calculations (finger-level)
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
# Wrist angle calculation (from wrist_angle_tracker.py)
# =========================================================================

def _calculate_wrist_angle(landmarks: Landmarks) -> float:
    """Compute wrist angle in image plane (2D: x, y). Returns degrees."""
    pts = np.array(landmarks)
    wrist = pts[HandLandmark.WRIST, :2]
    tip_center = np.mean(pts[TIP_INDICES, :2], axis=0)
    hand_vec = tip_center - wrist
    return float(np.degrees(np.arctan2(hand_vec[1], hand_vec[0])))


# =========================================================================
# Result data class
# =========================================================================

@dataclass
class StimulationMovementResult:
    """Complete movement result for one stimulation event."""

    timestamp: Optional[datetime] = None
    latency_ms: float = 0.0

    # Wrist
    wrist_angle_delta: float = 0.0
    wrist_direction: str = ""           # "upward", "downward", "neutral"
    wrist_feedback: str = ""            # "24.2° downward (flexion)"

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
    movement_type: str = ""             # "flexion", "extension", "mixed", "none"

    # Quality
    hand_detected: bool = False
    confidence: float = 0.0

    # Stimulation params (filled by caller)
    channel: Optional[int] = None
    intensity_ma: Optional[int] = None
    pulse_width_us: Optional[int] = None

    def to_dict(self) -> dict:
        """Export for future database storage."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latency_ms": self.latency_ms,
            "wrist_angle_delta": self.wrist_angle_delta,
            "wrist_direction": self.wrist_direction,
            "wrist_feedback": self.wrist_feedback,
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
# Main tracker class
# =========================================================================

class MovementTracker:
    """
    Webcam-based hand movement tracker for EMS experiments.

    Runs MediaPipe hand detection in a background thread.
    Provides simple methods to capture baseline and measure movement.

    Integration with EMS GUI:
        tracker = MovementTracker()
        tracker.start()                    # starts webcam thread

        # Before stimulation:
        tracker.capture_baseline()         # locks current hand pose

        # After stimulation:
        result = tracker.measure_movement()  # compares to baseline
        # → prints full report to terminal

        tracker.stop()                     # stops webcam thread
    """

    def __init__(
        self,
        camera_index: int = 0,
        smoothing_window: int = 5,
        noise_threshold: float = 2.0,
        model_path: Optional[str] = None,
        post_stim_delay: float = 0.5,
        show_preview: bool = True,
        save_dir: str = "captures",
    ):
        """
        Args:
            camera_index: Webcam index (0 = default).
            smoothing_window: Frames to average for noise reduction.
            noise_threshold: Min degrees to count as wrist movement.
            model_path: Path to hand_landmarker.task (auto-downloads if None).
            post_stim_delay: Seconds to wait after stim before measuring
                             (gives muscles time to respond).
            show_preview: Show live webcam preview window with hand overlay.
            save_dir: Directory to save baseline/result frame images.
        """
        self.camera_index = camera_index
        self.smoothing_window = smoothing_window
        self.noise_threshold = noise_threshold
        self.post_stim_delay = post_stim_delay
        self.show_preview = show_preview
        self.save_dir = save_dir

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stim_count = 0  # Counter for naming saved images

        # Latest frame data (updated by background thread)
        self._latest_landmarks: Optional[Landmarks] = None
        self._latest_confidence: float = 0.0
        self._hand_detected: bool = False
        self._preview_frame: Optional[np.ndarray] = None  # Annotated BGR frame for GUI
        self._raw_frame: Optional[np.ndarray] = None       # Raw BGR frame (no overlay)

        # Smoothing buffers
        self._wrist_angle_buffer: deque = deque(maxlen=smoothing_window)
        self._landmarks_buffer: deque = deque(maxlen=smoothing_window)

        # Baseline
        self._baseline_landmarks: Optional[Landmarks] = None
        self._baseline_wrist_angle: Optional[float] = None
        self._baseline_finger_angles: Dict[str, List[float]] = {}
        self._baseline_flexion: Dict[str, float] = {}
        self._baseline_timestamp: Optional[datetime] = None
        self._baseline_set: bool = False

        # Last measurement result (for preview display)
        self._last_result: Optional[StimulationMovementResult] = None

        # MediaPipe
        self._model_path = model_path
        self._detector = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Start the webcam and hand detection thread.

        Returns:
            True if started successfully.
        """
        if not MEDIAPIPE_AVAILABLE:
            print("✗ Cannot start tracker — MediaPipe not installed")
            return False

        # Download model if needed
        self._model_path = self._ensure_model(self._model_path)
        if self._model_path is None:
            print("✗ Cannot start tracker — model not available")
            return False

        # Initialize detector
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self._model_path),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

        # Start background thread
        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        print("✓ Movement tracker started (webcam + hand detection)")
        return True

    def stop(self) -> None:
        """Stop the webcam and detection thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._detector:
            self._detector.close()
            self._detector = None

        # Clean up preview window
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self._baseline_landmarks = None
        self._baseline_wrist_angle = None
        self._baseline_set = False
        self._last_result = None
        print("✓ Movement tracker stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def hand_detected(self) -> bool:
        with self._lock:
            return self._hand_detected

    # ------------------------------------------------------------------
    # Baseline & Measurement (called from main_window.py)
    # ------------------------------------------------------------------

    def capture_baseline(self) -> bool:
        """
        Lock the current hand pose as baseline.
        Call this right before EMS stimulation.

        Returns:
            True if baseline captured (hand was detected).
        """
        with self._lock:
            if not self._hand_detected or self._latest_landmarks is None:
                print("✗ Cannot capture baseline — no hand detected")
                return False

            # Use averaged landmarks from buffer for stability
            if len(self._landmarks_buffer) >= 3:
                avg_landmarks = self._average_landmarks(list(self._landmarks_buffer))
            else:
                avg_landmarks = self._latest_landmarks

            self._baseline_landmarks = avg_landmarks
            self._baseline_timestamp = datetime.now()

            # Pre-compute baseline metrics
            self._baseline_finger_angles = _calculate_finger_angles(avg_landmarks)
            self._baseline_flexion = _calculate_total_flexion(self._baseline_finger_angles)

            # Wrist angle (averaged from buffer)
            if len(self._wrist_angle_buffer) >= 2:
                self._baseline_wrist_angle = float(np.mean(list(self._wrist_angle_buffer)))
            else:
                self._baseline_wrist_angle = _calculate_wrist_angle(avg_landmarks)

            self._baseline_set = True
            self._last_result = None  # Clear previous result

        self._stim_count += 1

        # Save baseline frame
        path = self._save_frame("baseline", self._baseline_landmarks,
                                extra_text="BASELINE — before stimulation")
        if path:
            print(f"  📸 Baseline frame saved: {path}")

        print(f"✓ Baseline captured at {self._baseline_timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        return True

    def measure_movement(
        self,
        recording_duration: float = 2.0,
        channel: Optional[int] = None,
        intensity: Optional[int] = None,
        pulse_width: Optional[int] = None,
    ) -> Optional[StimulationMovementResult]:
        """
        Record frames for a duration and find the PEAK movement from baseline.

        Instead of capturing a single frame after a fixed delay, this method:
        1. Records all frames for `recording_duration` seconds
        2. Computes wrist angle + finger angles for every frame
        3. Finds the frame with maximum total deviation from baseline
        4. Reports that peak frame as the result

        This captures the actual muscle contraction even if it happens
        mid-stimulation and relaxes quickly afterward.

        Args:
            recording_duration: How long to record in seconds.
                                Should cover the full stimulation + a short tail.
                                Default 2.0s covers most 200ms–1s pulses.
            channel: EMS channel (for logging).
            intensity: EMS intensity in mA (for logging).
            pulse_width: EMS pulse width in μs (for logging).

        Returns:
            StimulationMovementResult from the peak frame, or None.
        """
        if self._baseline_landmarks is None:
            print("✗ Cannot measure — no baseline set")
            return None

        print(f"  ⏺ Recording for {recording_duration}s (capturing peak movement)...")

        # --- Collect frames for the recording window ---
        recorded_frames: List[dict] = []
        start_time = time.time()

        while (time.time() - start_time) < recording_duration:
            with self._lock:
                if not self._hand_detected or self._latest_landmarks is None:
                    time.sleep(0.01)
                    continue

                landmarks = list(self._latest_landmarks)  # copy
                confidence = self._latest_confidence

            elapsed_ms = (time.time() - start_time) * 1000

            # Compute metrics for this frame
            wrist_angle = _calculate_wrist_angle(landmarks)
            finger_angles = _calculate_finger_angles(landmarks)
            flexion = _calculate_total_flexion(finger_angles)

            # Compute total deviation from baseline
            wrist_delta_raw = wrist_angle - self._baseline_wrist_angle
            wrist_delta = (wrist_delta_raw + 180) % 360 - 180

            flexion_change = {
                f: flexion.get(f, 0) - self._baseline_flexion.get(f, 0)
                for f in FINGER_NAMES
            }
            total_finger_dev = sum(abs(v) for v in flexion_change.values())

            # Combined score: wrist + fingers (weighted)
            total_deviation = abs(wrist_delta) + total_finger_dev

            recorded_frames.append({
                "landmarks": landmarks,
                "confidence": confidence,
                "elapsed_ms": elapsed_ms,
                "wrist_angle": wrist_angle,
                "wrist_delta": wrist_delta,
                "finger_angles": finger_angles,
                "flexion": flexion,
                "flexion_change": flexion_change,
                "total_finger_dev": total_finger_dev,
                "total_deviation": total_deviation,
            })

            time.sleep(0.02)  # ~50 FPS sampling

        if not recorded_frames:
            print("✗ No frames captured during recording window")
            return None

        print(f"  ✓ Recorded {len(recorded_frames)} frames over {recording_duration}s")

        # --- Find peak frame (maximum total deviation from baseline) ---
        peak_frame = max(recorded_frames, key=lambda f: f["total_deviation"])
        peak_idx = recorded_frames.index(peak_frame)

        print(f"  Peak movement at frame {peak_idx + 1}/{len(recorded_frames)} "
              f"({peak_frame['elapsed_ms']:.0f}ms after stim)")

        # --- Build result from peak frame ---
        current_landmarks = peak_frame["landmarks"]
        wrist_delta = peak_frame["wrist_delta"]
        result_finger_angles = peak_frame["finger_angles"]
        result_flexion = peak_frame["flexion"]
        flexion_change = peak_frame["flexion_change"]
        current_confidence = peak_frame["confidence"]

        # Wrist classification
        if abs(wrist_delta) < self.noise_threshold:
            wrist_direction = "neutral"
            wrist_feedback = f"{abs(wrist_delta):.1f}° — no significant wrist movement"
        elif wrist_delta < 0:
            wrist_direction = "upward"
            wrist_feedback = f"{abs(wrist_delta):.1f}° upward (extension)"
        else:
            wrist_direction = "downward"
            wrist_feedback = f"{abs(wrist_delta):.1f}° downward (flexion)"

        # Finger angle deltas
        angle_deltas = {}
        for finger in FINGER_NAMES:
            b = self._baseline_finger_angles.get(finger, [0, 0, 0])
            r = result_finger_angles.get(finger, [0, 0, 0])
            angle_deltas[finger] = [rv - bv for bv, rv in zip(b, r)]

        # Total movement
        total_movement = sum(sum(abs(d) for d in deltas) for deltas in angle_deltas.values())

        # Primary finger
        primary_finger = max(flexion_change, key=lambda f: abs(flexion_change[f]))
        primary_movement = flexion_change[primary_finger]

        # Classify
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

        now = datetime.now()
        latency_ms = peak_frame["elapsed_ms"]

        result = StimulationMovementResult(
            timestamp=now,
            latency_ms=latency_ms,
            wrist_angle_delta=round(wrist_delta, 1),
            wrist_direction=wrist_direction,
            wrist_feedback=wrist_feedback,
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
            confidence=current_confidence,
            channel=channel,
            intensity_ma=intensity,
            pulse_width_us=pulse_width,
        )

        # Print to terminal
        self._print_result(result)
        self._last_result = result

        # Save result frame with measurement overlay
        result_text = (
            f"PEAK at {latency_ms:.0f}ms (frame {peak_idx + 1}/{len(recorded_frames)})\n"
            f"Wrist: {result.wrist_feedback}\n"
            f"Primary: {result.primary_finger} ({result.primary_finger_movement:+.1f}°)\n"
            f"Total finger movement: {result.total_finger_movement:.1f}°\n"
            f"Type: {result.movement_type}"
        )
        path = self._save_frame("peak", current_landmarks, extra_text=result_text)
        if path:
            print(f"  📸 Peak frame saved: {path}")
            print(f"  📁 All captures in: {os.path.abspath(self.save_dir)}/")

        # Also print the movement timeline
        self._print_timeline(recorded_frames)

        return result

    def _print_timeline(self, frames: List[dict]) -> None:
        """Print a mini timeline showing movement magnitude over time."""
        if not frames:
            return

        max_dev = max(f["total_deviation"] for f in frames)
        if max_dev < 1:
            return

        print()
        print("  --- Movement Timeline ---")
        # print(f"  {'Time':>6s}  {'Wrist':>7s}  {'Fingers':>8s}  {'Total':>7s}  Graph")
        # print(f"  {'-'*55}")

        # Sample ~15 points across the timeline
        n = len(frames)
        step = max(1, n // 15)

        for i in range(0, n, step):
            f = frames[i]
            bar_len = int((f["total_deviation"] / max_dev) * 25)
            bar = "█" * bar_len

            is_peak = (f["total_deviation"] == max_dev)
            marker = "" if is_peak else ""

            print(f"  {f['elapsed_ms']:5.0f}ms  {f['wrist_delta']:+6.1f}°  "
                  f"{f['total_finger_dev']:+7.1f}°  {f['total_deviation']:6.1f}  "
                  f"{bar}{marker}")

        print()

    def capture_and_measure(
        self,
        channel: Optional[int] = None,
        intensity: Optional[int] = None,
        pulse_width: Optional[int] = None,
        pre_stim_settle: float = 0.3,
    ) -> Optional[StimulationMovementResult]:
        """
        All-in-one: capture baseline → wait → measure.

        This is NOT for auto-integration with the STIMULATE button.
        Use capture_baseline() + measure_movement() for that.

        This is for manual testing without EMS.
        """
        print("\n=== Manual Capture & Measure ===")
        print(f"  Settling for {pre_stim_settle}s...")
        time.sleep(pre_stim_settle)

        if not self.capture_baseline():
            return None

        return self.measure_movement(channel, intensity, pulse_width)

    # ------------------------------------------------------------------
    # Background camera thread
    # ------------------------------------------------------------------

    # Hand skeleton connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    def _camera_loop(self) -> None:
        """
        Background thread: reads webcam and runs hand detection.

        Does NOT call cv2.imshow (crashes on macOS from non-main thread).
        Instead, stores the annotated frame in self._preview_frame so
        the GUI can display it via a QTimer + QLabel.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("✗ Could not open webcam")
            self._running = False
            return

        print(f"  Webcam opened (camera {self.camera_index})") 

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Detect hand
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)

            landmarks_px = None

            with self._lock:
                if result.hand_landmarks:
                    hand_lm = result.hand_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lm]
                    landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

                    self._latest_landmarks = landmarks
                    self._hand_detected = True
                    self._landmarks_buffer.append(landmarks)

                    wrist_angle = _calculate_wrist_angle(landmarks)
                    self._wrist_angle_buffer.append(wrist_angle)

                    if result.handedness:
                        self._latest_confidence = result.handedness[0][0].score
                else:
                    self._hand_detected = False

            # Store raw frame (before overlay) for saving
            with self._lock:
                self._raw_frame = frame.copy()

            # Draw annotations on frame and store for GUI to pick up
            if self.show_preview:
                self._draw_preview(frame, landmarks_px, result)
                with self._lock:
                    self._preview_frame = frame.copy()
            else:
                with self._lock:
                    self._preview_frame = None

        cap.release()
        print("  Webcam released")

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest annotated frame (BGR) for display in the GUI.

        Called by a QTimer in the GUI to update a QLabel.

        Returns:
            BGR frame with hand overlay, or None if not available.
        """
        with self._lock:
            if self._preview_frame is not None:
                return self._preview_frame.copy()
        return None

    def _draw_preview(self, frame, landmarks_px, detection_result) -> None:
        """Draw hand skeleton, status, and last result on frame."""
        h, w = frame.shape[:2]

        # --- Draw hand skeleton ---
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

            # Handedness
            if detection_result.handedness:
                label = detection_result.handedness[0][0].category_name
                conf = detection_result.handedness[0][0].score
                cv2.putText(frame, f"{label} ({conf:.0%})", (w - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Status bar at top ---
        status_color = (0, 255, 0) if self._hand_detected else (0, 0, 255)
        status_text = "HAND DETECTED" if self._hand_detected else "NO HAND"
        cv2.putText(frame, status_text, (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Baseline status
        if self._baseline_set:
            cv2.putText(frame, "BASELINE SET", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        else:
            cv2.putText(frame, "Waiting for stimulation...", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # --- Last result overlay ---
        if self._last_result and self._last_result.hand_detected:
            r = self._last_result
            y0 = h - 140

            cv2.rectangle(frame, (10, y0 - 10), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, y0 - 10), (w - 10, h - 10), (255, 255, 255), 1)

            # Wrist result
            wrist_color = (0, 100, 255) if r.wrist_direction == "downward" else \
                          (0, 255, 100) if r.wrist_direction == "upward" else (200, 200, 200)
            cv2.putText(frame, f"Wrist: {r.wrist_feedback}", (20, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, wrist_color, 2)

            # Finger summary
            cv2.putText(frame, f"Fingers: {r.primary_finger} {r.primary_finger_movement:+.1f}°  |  "
                        f"Total: {r.total_finger_movement:.1f}°  |  Type: {r.movement_type}",
                        (20, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Per-finger deltas
            x_offset = 20
            y_line = y0 + 80
            for finger in FINGER_NAMES:
                change = r.flexion_change.get(finger, 0)
                color = (100, 255, 100) if abs(change) > 10 else (150, 150, 150)
                text = f"{finger[:3]}:{change:+.0f}°"
                cv2.putText(frame, text, (x_offset, y_line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                x_offset += 100

            # Stim params
            if r.channel is not None:
                cv2.putText(frame, f"Ch{r.channel} {r.intensity_ma}mA {r.pulse_width_us}μs",
                            (20, y0 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_frame(self, label: str, landmarks: Optional[Landmarks] = None,
                    extra_text: Optional[str] = None) -> Optional[str]:
        """
        Save the current raw frame with landmarks drawn on it.

        Args:
            label: Label for the file (e.g. "baseline", "result").
            landmarks: Landmarks to draw. If None, draws from latest.
            extra_text: Extra text to overlay on the image.

        Returns:
            Path to saved image, or None if no frame available.
        """
        with self._lock:
            if self._raw_frame is None:
                return None
            frame = self._raw_frame.copy()

        h, w = frame.shape[:2]

        # Draw landmarks if provided
        if landmarks is not None:
            landmarks_px = [(int(x * w), int(y * h)) for x, y, z in landmarks]

            # Skeleton
            for s, e in self.HAND_CONNECTIONS:
                cv2.line(frame, landmarks_px[s], landmarks_px[e], (255, 255, 255), 2)

            # Joints
            for i, (px, py) in enumerate(landmarks_px):
                if i == 0:
                    color = (0, 0, 255)       # Wrist
                elif i in [4, 8, 12, 16, 20]:
                    color = (0, 255, 0)       # Fingertips
                else:
                    color = (255, 0, 0)       # Other joints
                cv2.circle(frame, (px, py), 6, color, -1)
                cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)

                # Label index
                cv2.putText(frame, str(i), (px + 8, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Label overlay
        cv2.putText(frame, label.upper(), (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(frame, timestamp_str, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if extra_text:
            # Draw text with background
            y_pos = h - 20
            for line in reversed(extra_text.strip().split("\n")):
                cv2.putText(frame, line, (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos -= 22

        # Save
        filename = f"stim{self._stim_count:03d}_{label}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)

        return filepath

    def _average_landmarks(self, landmarks_list: List[Landmarks]) -> Landmarks:
        """Average multiple frames of landmarks for stability."""
        arr = np.array(landmarks_list)  # (N, 21, 3)
        avg = np.mean(arr, axis=0)      # (21, 3)
        return [tuple(row) for row in avg]

    def _ensure_model(self, path: Optional[str]) -> Optional[str]:
        """Download hand landmarker model if needed."""
        if path and os.path.exists(path):
            return path

        default_path = "hand_landmarker.task"
        if os.path.exists(default_path):
            return default_path

        print("  Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, default_path)
            print(f"  ✓ Model downloaded: {default_path}")
            return default_path
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return None

    def _print_result(self, r: StimulationMovementResult) -> None:
        """Print full movement report to terminal."""
        print()
        print("=" * 60)
        print("  EMS STIMULATION — MOVEMENT RESULT")
        print("=" * 60)

        if r.channel is not None:
            print(f"  Stimulation: Ch{r.channel}, {r.intensity_ma}mA, {r.pulse_width_us}μs")

        print(f"  Latency: {r.latency_ms:.0f} ms")
        print(f"  Confidence: {r.confidence:.0%}")
        print()

        # Wrist
        print("  --- Wrist ---")
        print(f"  {r.wrist_feedback}")
        print()

        # Fingers
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
        print(f"  Primary finger: {r.primary_finger} ({r.primary_finger_movement:+.1f}°)")
        print(f"  Movement type: {r.movement_type}")
        print("=" * 60)
        print()
