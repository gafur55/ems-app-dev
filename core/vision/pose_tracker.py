"""
Pose tracker for EMS stimulation experiments.

Runs the laptop webcam in a background thread and provides:
- Wrist angle tracking using MediaPipe Pose Landmarker
- Calculates wrist flexion/extension as the angle at the WRIST joint
  formed by two vectors: ELBOW→WRIST and WRIST→INDEX_FINGER
- Automatic baseline capture before stimulation
- Automatic peak measurement after stimulation

This is the WRIST-ONLY tracker. Finger angles are handled by HandTracker.

Wrist angle uses a SIGNED angle calculation:
    v1 = elbow - wrist   (forearm vector)
    v2 = index - wrist   (hand vector)
    unsigned_angle = arccos(dot(v1, v2) / (|v1| * |v2|))
    sign = cross(v1, v2)   → positive = hand is BELOW forearm line
                             → negative = hand is ABOVE forearm line

    In image coordinates (y increases downward):
        cross > 0 means v2 is clockwise from v1 → hand points down → flexion
        cross < 0 means v2 is counter-clockwise  → hand points up → extension

    Convention:
        - Bending UP (extension) → POSITIVE delta
        - Bending DOWN (flexion) → NEGATIVE delta

Arm detection:
    By default, the tracker auto-detects which arm is more visible and
    tracks that one. You can also lock it to "left" or "right" with set_arm().

Pose Landmark indices used:
    Left arm:  elbow=13, wrist=15, index=19, pinky=17
    Right arm: elbow=14, wrist=16, index=20, pinky=18

Usage in main_window.py:
    from core.vision.pose_tracker import PoseTracker

    # On session start (auto-detect arm)
    self.pose_tracker = PoseTracker(camera_index=1)
    self.pose_tracker.start()

    # Before stimulation:
    self.pose_tracker.capture_baseline()
    # ... EMS fires ...
    result = self.pose_tracker.measure_movement()

    # On session stop
    self.pose_tracker.stop()
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
# Pose landmark indices (upper body only)
# =========================================================================

class PoseLandmark(IntEnum):
    """MediaPipe Pose landmark indices relevant to wrist tracking."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22


# Arm landmark sets keyed by side
ARM_LANDMARKS = {
    "left": {
        "shoulder": PoseLandmark.LEFT_SHOULDER,
        "elbow": PoseLandmark.LEFT_ELBOW,
        "wrist": PoseLandmark.LEFT_WRIST,
        "index": PoseLandmark.LEFT_INDEX,
        "pinky": PoseLandmark.LEFT_PINKY,
        "thumb": PoseLandmark.LEFT_THUMB,
    },
    "right": {
        "shoulder": PoseLandmark.RIGHT_SHOULDER,
        "elbow": PoseLandmark.RIGHT_ELBOW,
        "wrist": PoseLandmark.RIGHT_WRIST,
        "index": PoseLandmark.RIGHT_INDEX,
        "pinky": PoseLandmark.RIGHT_PINKY,
        "thumb": PoseLandmark.RIGHT_THUMB,
    },
}

# Pose skeleton connections for drawing (upper body subset)
UPPER_BODY_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (15, 17), (15, 19), (15, 21),  # left hand
    (16, 18), (16, 20), (16, 22),  # right hand
    (17, 19), (18, 20),  # pinky-index connections
]

# Type alias
PoseLandmarks = List[Tuple[float, float, float]]


# =========================================================================
# Signed angle calculation
# =========================================================================

def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the UNSIGNED angle at p2 formed by p1→p2 and p2→p3.

    Returns:
        Angle in degrees (0-180).
    """
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _calculate_signed_wrist_angle(
    elbow: np.ndarray, wrist: np.ndarray, index: np.ndarray
) -> float:
    """
    Calculate the SIGNED wrist angle using the 2D cross product.

    Convention (after accounting for image y-axis pointing down):
        POSITIVE angle  ->  extension (hand bends UP in real world)
        NEGATIVE angle  ->  flexion  (hand bends DOWN in real world)
        ~0 degrees      ->  straight wrist

    The sign comes from the 2D cross product of:
        v1 = elbow - wrist  (forearm direction, pointing toward elbow)
        v2 = index - wrist  (hand direction, pointing toward fingertips)

    In IMAGE coordinates (y-down):
        cross(v1, v2) > 0  ->  v2 is clockwise from v1  ->  hand is BELOW forearm
        cross(v1, v2) < 0  ->  v2 is counter-clockwise   ->  hand is ABOVE forearm

    We want: up = positive, down = negative.
    So: cross > 0 (hand below) = flexion = NEGATIVE
        cross < 0 (hand above) = extension = POSITIVE

    Returns:
        Signed deviation from straight wrist in degrees.
        Straight wrist ~ 0. Extension > 0. Flexion < 0.
    """
    v1 = elbow - wrist   # forearm direction
    v2 = index - wrist   # hand direction

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0

    # Unsigned angle (0-180)
    cos_a = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    unsigned_angle = float(np.degrees(np.arccos(cos_a)))

    # Deviation from straight (180 = straight)
    deviation = 180.0 - unsigned_angle

    # 2D cross product: v1.x * v2.y - v1.y * v2.x
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])

    # In image coords (y-down) with MIRRORED frame (cv2.flip horizontally):
    #   The mirror reverses x, which flips the cross product sign.
    #   cross > 0 -> extension (hand UP)  -> POSITIVE
    #   cross < 0 -> flexion  (hand DOWN) -> NEGATIVE
    if cross > 0:
        signed_deviation = deviation   # extension (up)
    elif cross < 0:
        signed_deviation = -deviation  # flexion (down)
    else:
        signed_deviation = 0.0

    return signed_deviation


# =========================================================================
# Result data class
# =========================================================================

@dataclass
class WristMovementResult:
    """Movement result for wrist angle from one stimulation event."""

    timestamp: Optional[datetime] = None
    latency_ms: float = 0.0

    # Wrist angle
    baseline_wrist_angle: float = 0.0
    result_wrist_angle: float = 0.0
    wrist_angle_delta: float = 0.0
    wrist_direction: str = ""       # "flexion", "extension", "neutral"
    wrist_feedback: str = ""        # e.g., "24.2 degrees flexion"

    # Quality
    pose_detected: bool = False
    visibility: float = 0.0        # average visibility of arm landmarks

    # Stimulation params (filled by caller)
    channel: Optional[int] = None
    intensity_ma: Optional[int] = None
    pulse_width_us: Optional[int] = None

    # Arm side
    arm_side: str = ""

    def to_dict(self) -> dict:
        """Export for database storage."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latency_ms": self.latency_ms,
            "baseline_wrist_angle": self.baseline_wrist_angle,
            "result_wrist_angle": self.result_wrist_angle,
            "wrist_angle_delta": self.wrist_angle_delta,
            "wrist_direction": self.wrist_direction,
            "wrist_feedback": self.wrist_feedback,
            "pose_detected": self.pose_detected,
            "visibility": self.visibility,
            "channel": self.channel,
            "intensity_ma": self.intensity_ma,
            "pulse_width_us": self.pulse_width_us,
            "arm_side": self.arm_side,
        }


# =========================================================================
# Pose Tracker class
# =========================================================================

class PoseTracker:
    """
    Webcam-based wrist angle tracker for EMS experiments.

    Uses MediaPipe Pose Landmarker on the laptop camera.
    Calculates wrist angle as the signed deviation from a straight wrist.

    Arm selection:
        - "auto" (default): picks the arm with better visibility each frame
        - "left" / "right": locks to one arm

    Does NOT track finger joint angles -- that is handled by HandTracker.
    """

    def __init__(
        self,
        camera_index: int = 1,
        arm: str = "auto",
        smoothing_window: int = 5,
        noise_threshold: float = 3.0,
        model_path: Optional[str] = None,
        show_preview: bool = True,
        save_dir: str = "captures/pose",
    ):
        """
        Args:
            camera_index: Webcam index (1 = laptop on Mac with Continuity Camera).
            arm: Which arm to track -- "left", "right", or "auto".
            smoothing_window: Frames to average for noise reduction.
            noise_threshold: Min degrees to count as wrist movement.
            model_path: Path to pose_landmarker.task (auto-downloads if None).
            show_preview: Store annotated preview frames for GUI display.
            save_dir: Directory to save baseline/result frame images.
        """
        self.camera_index = camera_index
        self.smoothing_window = smoothing_window
        self.noise_threshold = noise_threshold
        self.show_preview = show_preview
        self.save_dir = save_dir

        # Arm selection
        arm = arm.lower()
        if arm not in ("left", "right", "auto"):
            raise ValueError(f"arm must be 'left', 'right', or 'auto', got '{arm}'")
        self._arm_mode = arm               # "auto", "left", or "right"
        self.arm = arm if arm != "auto" else "left"  # current active arm
        self._arm_lm = ARM_LANDMARKS[self.arm]

        os.makedirs(self.save_dir, exist_ok=True)

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stim_count = 0

        # Latest frame data (updated by background thread)
        self._latest_landmarks: Optional[PoseLandmarks] = None
        self._latest_visibility: float = 0.0
        self._pose_detected: bool = False
        self._preview_frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None

        # Smoothing buffers
        self._wrist_angle_buffer: deque = deque(maxlen=smoothing_window)
        self._landmarks_buffer: deque = deque(maxlen=smoothing_window)

        # Baseline
        self._baseline_landmarks: Optional[PoseLandmarks] = None
        self._baseline_wrist_angle: Optional[float] = None
        self._baseline_timestamp: Optional[datetime] = None
        self._baseline_set: bool = False

        # Last result (for preview overlay)
        self._last_result: Optional[WristMovementResult] = None

        # MediaPipe
        self._model_path = model_path
        self._detector = None
        self._undistorter: Optional[Undistorter] = None

    # ------------------------------------------------------------------
    # Public: arm selection
    # ------------------------------------------------------------------

    def set_arm(self, arm: str) -> None:
        """
        Change which arm to track.

        Args:
            arm: "left", "right", or "auto"
        """
        arm = arm.lower()
        if arm not in ("left", "right", "auto"):
            raise ValueError(f"arm must be 'left', 'right', or 'auto', got '{arm}'")
        self._arm_mode = arm
        if arm != "auto":
            self.arm = arm
            self._arm_lm = ARM_LANDMARKS[self.arm]
        print(f"✓ [Pose] Arm mode: {self._arm_mode}")

    def _auto_select_arm(self, landmarks: PoseLandmarks) -> str:
        """
        Pick the arm with better average visibility across elbow, wrist, index.

        Returns:
            "left" or "right"
        """
        def _arm_visibility(side: str) -> float:
            lm_set = ARM_LANDMARKS[side]
            total = 0.0
            for key in ("elbow", "wrist", "index", "pinky"):
                idx = lm_set[key]
                lm = landmarks[idx]
                total += lm[3] if len(lm) > 3 else 1.0
            return total / 4.0

        left_vis = _arm_visibility("left")
        right_vis = _arm_visibility("right")
        return "left" if left_vis >= right_vis else "right"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the camera and pose detection thread."""
        if not MEDIAPIPE_AVAILABLE:
            print("✗ Cannot start pose tracker -- MediaPipe not installed")
            return False

        # Load calibration — raises CalibrationNotFoundError if missing
        self._undistorter = Undistorter(self.camera_index)

        self._model_path = self._ensure_model(self._model_path)
        if self._model_path is None:
            print("✗ Cannot start pose tracker -- model not available")
            return False

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self._model_path),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)

        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        arm_desc = self._arm_mode if self._arm_mode != "auto" else "auto-detect"
        print(f"✓ Pose tracker started (camera {self.camera_index} -- laptop, "
              f"arm: {arm_desc})")
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
        print("✓ Pose tracker stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pose_detected(self) -> bool:
        with self._lock:
            return self._pose_detected

    # ------------------------------------------------------------------
    # Wrist angle from pose landmarks (SIGNED)
    # ------------------------------------------------------------------

    def _calculate_wrist_angle(self, landmarks: PoseLandmarks) -> float:
        """
        Calculate SIGNED wrist angle from pose landmarks.

        Uses midpoint of INDEX and PINKY as the hand direction point
        for a more stable center-of-hand vector.

        Positive = extension (hand up), Negative = flexion (hand down).
        A straight wrist ~ 0 degrees deviation.

        Returns:
            Signed deviation from straight wrist in degrees.
        """
        elbow_idx = self._arm_lm["elbow"]
        wrist_idx = self._arm_lm["wrist"]
        index_idx = self._arm_lm["index"]
        pinky_idx = self._arm_lm["pinky"]

        elbow = np.array(landmarks[elbow_idx][:2])  # x, y in image coords
        wrist = np.array(landmarks[wrist_idx][:2])
        index = np.array(landmarks[index_idx][:2])
        pinky = np.array(landmarks[pinky_idx][:2])

        # Midpoint of index and pinky = center of hand
        hand_center = (index + pinky) / 2.0

        return _calculate_signed_wrist_angle(elbow, wrist, hand_center)

    def _calculate_wrist_angle_unsigned(self, landmarks: PoseLandmarks) -> float:
        """Unsigned wrist angle (0-180) for display purposes."""
        elbow_idx = self._arm_lm["elbow"]
        wrist_idx = self._arm_lm["wrist"]
        index_idx = self._arm_lm["index"]
        pinky_idx = self._arm_lm["pinky"]

        elbow = np.array(landmarks[elbow_idx][:2])
        wrist = np.array(landmarks[wrist_idx][:2])
        index = np.array(landmarks[index_idx][:2])
        pinky = np.array(landmarks[pinky_idx][:2])

        hand_center = (index + pinky) / 2.0

        return _calculate_angle(elbow, wrist, hand_center)

    def _get_arm_visibility(self, landmarks: PoseLandmarks) -> float:
        """Average visibility of the arm landmarks used (elbow, wrist, index, pinky)."""
        total = 0.0
        count = 0
        for key in ("elbow", "wrist", "index", "pinky"):
            idx = self._arm_lm[key]
            lm = landmarks[idx]
            if len(lm) > 3:
                total += lm[3]
            else:
                total += 1.0
            count += 1
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Baseline & Measurement
    # ------------------------------------------------------------------

    def capture_baseline(self) -> bool:
        """
        Lock the current wrist angle as baseline.
        Call right before EMS stimulation.

        Returns:
            True if baseline captured.
        """
        with self._lock:
            if not self._pose_detected or self._latest_landmarks is None:
                print("✗ [Pose] Cannot capture baseline -- no pose detected")
                return False

            if len(self._landmarks_buffer) >= 3:
                avg_landmarks = self._average_landmarks(list(self._landmarks_buffer))
            else:
                avg_landmarks = self._latest_landmarks

            self._baseline_landmarks = avg_landmarks
            self._baseline_timestamp = datetime.now()

            # Use averaged wrist angle from buffer for stability
            if len(self._wrist_angle_buffer) >= 2:
                self._baseline_wrist_angle = float(np.mean(list(self._wrist_angle_buffer)))
            else:
                self._baseline_wrist_angle = self._calculate_wrist_angle(avg_landmarks)

            self._baseline_set = True
            self._last_result = None

        self._stim_count += 1

        # Save baseline frame
        path = self._save_frame("baseline",
                                extra_text=f"BASELINE -- wrist deviation: "
                                           f"{self._baseline_wrist_angle:+.1f} deg")
        if path:
            print(f"  [Pose] Baseline frame saved: {path}")

        print(f"✓ [Pose] Baseline captured: wrist deviation = "
              f"{self._baseline_wrist_angle:+.1f} deg "
              f"({self.arm} arm) at "
              f"{self._baseline_timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        return True

    def measure_movement(
        self,
        recording_duration: float = 2.0,
        channel: Optional[int] = None,
        intensity: Optional[int] = None,
        pulse_width: Optional[int] = None,
    ) -> Optional[WristMovementResult]:
        """
        Record frames and find PEAK wrist angle change from baseline.

        Args:
            recording_duration: How long to record in seconds.
            channel: EMS channel (for logging).
            intensity: EMS intensity in mA (for logging).
            pulse_width: EMS pulse width in us (for logging).

        Returns:
            WristMovementResult from the peak frame, or None.
        """
        if self._baseline_wrist_angle is None:
            print("✗ [Pose] Cannot measure -- no baseline set")
            return None

        print(f"  [Pose] Recording for {recording_duration}s...")

        # Collect frames
        recorded_frames: List[dict] = []
        frame_snapshots: dict = {}  # index -> raw frame for saving
        start_time = time.time()
        sample_count = 0

        while (time.time() - start_time) < recording_duration:
            with self._lock:
                if not self._pose_detected or self._latest_landmarks is None:
                    time.sleep(0.01)
                    continue
                landmarks = list(self._latest_landmarks)
                visibility = self._latest_visibility
                # Save raw frame every 5th sample for peak image
                if sample_count % 5 == 0 and self._raw_frame is not None:
                    frame_snapshots[len(recorded_frames)] = self._raw_frame.copy()

            elapsed_ms = (time.time() - start_time) * 1000

            wrist_angle = self._calculate_wrist_angle(landmarks)
            wrist_delta = wrist_angle - self._baseline_wrist_angle

            recorded_frames.append({
                "landmarks": landmarks,
                "visibility": visibility,
                "elapsed_ms": elapsed_ms,
                "wrist_angle": wrist_angle,
                "wrist_delta": wrist_delta,
                "abs_delta": abs(wrist_delta),
            })

            sample_count += 1
            time.sleep(0.02)  # ~50 FPS sampling

        if not recorded_frames:
            print("✗ [Pose] No frames captured during recording window")
            return None

        print(f"  ✓ [Pose] Recorded {len(recorded_frames)} frames")

        # Find peak region: average frames within 80% of max displacement
        max_delta = max(f["abs_delta"] for f in recorded_frames)
        threshold = max_delta * 0.8

        near_peak_frames = [f for f in recorded_frames if f["abs_delta"] >= threshold]

        print(f"  [Pose] Max delta: {max_delta:.1f} deg, "
              f"threshold (80%): {threshold:.1f} deg, "
              f"near-peak frames: {len(near_peak_frames)}/{len(recorded_frames)}")

        # Average the near-peak frames
        avg_wrist_angle = float(np.mean([f["wrist_angle"] for f in near_peak_frames]))
        avg_wrist_delta = float(np.mean([f["wrist_delta"] for f in near_peak_frames]))
        avg_visibility = float(np.mean([f["visibility"] for f in near_peak_frames]))
        avg_elapsed_ms = float(np.mean([f["elapsed_ms"] for f in near_peak_frames]))

        # Use the single highest frame for the saved image (closest to true peak)
        peak_frame = max(near_peak_frames, key=lambda f: f["abs_delta"])
        peak_idx = recorded_frames.index(peak_frame)

        print(f"  [Pose] Averaged {len(near_peak_frames)} frames near peak "
              f"({avg_elapsed_ms:.0f}ms avg latency)")

        # Build result using averaged values
        wrist_delta = avg_wrist_delta
        result_angle = avg_wrist_angle

        # Classify direction using signed delta
        # Positive delta = extension (up), Negative delta = flexion (down)
        if abs(wrist_delta) < self.noise_threshold:
            wrist_direction = "neutral"
            wrist_feedback = f"{abs(wrist_delta):.1f} deg -- no significant wrist movement"
        elif wrist_delta > 0:
            wrist_direction = "extension"
            wrist_feedback = f"{abs(wrist_delta):.1f} deg extension (wrist bent up)"
        else:
            wrist_direction = "flexion"
            wrist_feedback = f"{abs(wrist_delta):.1f} deg flexion (wrist bent down)"

        result = WristMovementResult(
            timestamp=datetime.now(),
            latency_ms=avg_elapsed_ms,
            baseline_wrist_angle=round(self._baseline_wrist_angle, 1),
            result_wrist_angle=round(result_angle, 1),
            wrist_angle_delta=round(wrist_delta, 1),
            wrist_direction=wrist_direction,
            wrist_feedback=wrist_feedback,
            pose_detected=True,
            visibility=avg_visibility,
            channel=channel,
            intensity_ma=intensity,
            pulse_width_us=pulse_width,
            arm_side=self.arm,
        )

        self._print_result(result)
        self._last_result = result

        # Save peak frame from the closest stored snapshot
        if frame_snapshots:
            # Find closest snapshot index to peak
            closest_idx = min(frame_snapshots.keys(), key=lambda i: abs(i - peak_idx))
            peak_raw_frame = frame_snapshots[closest_idx]
            path = self._save_frame_from("peak", peak_raw_frame,
                                          peak_frame["landmarks"],
                                          extra_text=f"PEAK at {result.latency_ms:.0f}ms\n"
                                                     f"Wrist: {result.wrist_feedback}")
            if path:
                print(f"  [Pose] Peak frame saved: {path}")
        else:
            # Fallback: save current frame
            path = self._save_frame("peak",
                                    extra_text=f"PEAK at {result.latency_ms:.0f}ms\n"
                                               f"Wrist: {result.wrist_feedback}")
            if path:
                print(f"  [Pose] Peak frame saved: {path} (fallback - not exact peak)")

        return result

    # ------------------------------------------------------------------
    # Background camera thread
    # ------------------------------------------------------------------

    def _camera_loop(self) -> None:
        """Background thread: reads laptop webcam and runs pose detection."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"✗ [Pose] Could not open camera {self.camera_index}")
            self._running = False
            return

        print(f"  [Pose] Camera {self.camera_index} opened")

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

            with self._lock:
                self._raw_frame = frame.copy()

                if result.pose_landmarks:
                    pose_lm = result.pose_landmarks[0]
                    # Store as (x, y, z, visibility) tuples
                    landmarks = [
                        (lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0))
                        for lm in pose_lm
                    ]

                    self._latest_landmarks = landmarks
                    self._pose_detected = True
                    self._landmarks_buffer.append(landmarks)

                    # Auto-detect arm if in auto mode
                    if self._arm_mode == "auto":
                        best_arm = self._auto_select_arm(landmarks)
                        if best_arm != self.arm:
                            self.arm = best_arm
                            self._arm_lm = ARM_LANDMARKS[self.arm]

                    # Calculate and buffer wrist angle (signed)
                    wrist_angle = self._calculate_wrist_angle(landmarks)
                    self._wrist_angle_buffer.append(wrist_angle)

                    # Average visibility of arm landmarks
                    self._latest_visibility = self._get_arm_visibility(landmarks)
                else:
                    self._pose_detected = False

            # Draw preview
            if self.show_preview:
                self._draw_preview(frame, result, h, w)
                with self._lock:
                    self._preview_frame = frame.copy()
            else:
                with self._lock:
                    self._preview_frame = None

        cap.release()
        print("  [Pose] Camera released")

    # ------------------------------------------------------------------
    # Preview & saving
    # ------------------------------------------------------------------

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame for GUI display."""
        with self._lock:
            if self._preview_frame is not None:
                return self._preview_frame.copy()
        return None

    def _draw_preview(self, frame, result, h: int, w: int) -> None:
        """Draw pose skeleton and wrist angle info on frame."""
        landmarks_px = None

        if result.pose_landmarks:
            pose_lm = result.pose_landmarks[0]
            landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in pose_lm]

            # Draw upper body skeleton (grey)
            for s, e in UPPER_BODY_CONNECTIONS:
                if s < len(landmarks_px) and e < len(landmarks_px):
                    cv2.line(frame, landmarks_px[s], landmarks_px[e],
                             (200, 200, 200), 2)

            # Highlight the tracked arm
            elbow_idx = self._arm_lm["elbow"]
            wrist_idx = self._arm_lm["wrist"]
            index_idx = self._arm_lm["index"]
            pinky_idx = self._arm_lm["pinky"]

            # Calculate hand center (midpoint of index and pinky)
            hand_center = (
                (landmarks_px[index_idx][0] + landmarks_px[pinky_idx][0]) // 2,
                (landmarks_px[index_idx][1] + landmarks_px[pinky_idx][1]) // 2,
            )

            # Draw tracked arm in bright colors
            cv2.line(frame, landmarks_px[elbow_idx], landmarks_px[wrist_idx],
                     (0, 255, 0), 3)  # Forearm = green
            cv2.line(frame, landmarks_px[wrist_idx], hand_center,
                     (0, 255, 255), 3)  # Hand vector = yellow

            # Draw index-pinky spread line (thin, subtle)
            cv2.line(frame, landmarks_px[index_idx], landmarks_px[pinky_idx],
                     (100, 100, 0), 1)

            # Draw landmark points
            cv2.circle(frame, landmarks_px[elbow_idx], 8, (255, 0, 0), -1)   # Blue = elbow
            cv2.circle(frame, landmarks_px[wrist_idx], 8, (0, 0, 255), -1)   # Red = wrist
            cv2.circle(frame, hand_center, 8, (0, 255, 255), -1)             # Cyan = hand center

            # Labels
            cv2.putText(frame, "ELBOW",
                        (landmarks_px[elbow_idx][0] + 10, landmarks_px[elbow_idx][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, "WRIST",
                        (landmarks_px[wrist_idx][0] + 10, landmarks_px[wrist_idx][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "MID",
                        (hand_center[0] + 10, hand_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Draw signed wrist angle
            with self._lock:
                if self._latest_landmarks is not None:
                    signed_angle = self._calculate_wrist_angle(self._latest_landmarks)
                    unsigned_angle = self._calculate_wrist_angle_unsigned(self._latest_landmarks)
                    direction_arrow = "UP" if signed_angle > 1 else "DOWN" if signed_angle < -1 else "--"
                    cv2.putText(frame,
                                f"Wrist: {unsigned_angle:.1f} deg ({signed_angle:+.1f} {direction_arrow})",
                                (landmarks_px[wrist_idx][0] + 10,
                                 landmarks_px[wrist_idx][1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No pose detected", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Status bar
        arm_mode_str = f"{self.arm.upper()}" if self._arm_mode != "auto" else f"AUTO>{self.arm.upper()}"
        status_color = (0, 255, 0) if self._pose_detected else (0, 0, 255)
        status_text = f"POSE DETECTED ({arm_mode_str} arm)" if self._pose_detected else "NO POSE"
        cv2.putText(frame, f"[WRIST] {status_text}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if self._baseline_set:
            cv2.putText(frame, f"BASELINE: {self._baseline_wrist_angle:+.1f} deg",
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # Last result overlay
        if self._last_result and self._last_result.pose_detected:
            r = self._last_result
            y0 = h - 60
            cv2.rectangle(frame, (10, y0 - 10), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, y0 - 10), (w - 10, h - 10), (255, 255, 255), 1)

            wrist_color = (0, 100, 255) if r.wrist_direction == "flexion" else \
                          (0, 255, 100) if r.wrist_direction == "extension" else (200, 200, 200)
            cv2.putText(frame, f"Wrist: {r.wrist_feedback}", (20, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, wrist_color, 2)

    def _save_frame_from(self, label: str, frame: np.ndarray,
                         landmarks: PoseLandmarks,
                         extra_text: Optional[str] = None) -> Optional[str]:
        """Save a specific raw frame with arm overlay (for peak snapshots)."""
        frame = frame.copy()
        h, w = frame.shape[:2]

        if landmarks is not None:
            elbow_idx = self._arm_lm["elbow"]
            wrist_idx = self._arm_lm["wrist"]
            index_idx = self._arm_lm["index"]
            pinky_idx = self._arm_lm["pinky"]

            pts = {
                "elbow": (int(landmarks[elbow_idx][0] * w), int(landmarks[elbow_idx][1] * h)),
                "wrist": (int(landmarks[wrist_idx][0] * w), int(landmarks[wrist_idx][1] * h)),
                "index": (int(landmarks[index_idx][0] * w), int(landmarks[index_idx][1] * h)),
                "pinky": (int(landmarks[pinky_idx][0] * w), int(landmarks[pinky_idx][1] * h)),
            }
            hand_center = (
                (pts["index"][0] + pts["pinky"][0]) // 2,
                (pts["index"][1] + pts["pinky"][1]) // 2,
            )

            cv2.line(frame, pts["elbow"], pts["wrist"], (0, 255, 0), 3)
            cv2.line(frame, pts["wrist"], hand_center, (0, 255, 255), 3)
            cv2.line(frame, pts["index"], pts["pinky"], (100, 100, 0), 1)

            cv2.circle(frame, pts["elbow"], 8, (255, 0, 0), -1)
            cv2.circle(frame, pts["wrist"], 8, (0, 0, 255), -1)
            cv2.circle(frame, hand_center, 8, (0, 255, 255), -1)

            for name, pt in [("ELBOW", pts["elbow"]), ("WRIST", pts["wrist"]), ("MID", hand_center)]:
                cv2.putText(frame, name, (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, f"[POSE] {label.upper()}", (15, 30),
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

        filename = f"pose_stim{self._stim_count:03d}_{label}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def _save_frame(self, label: str, extra_text: Optional[str] = None) -> Optional[str]:
        """Save the current raw frame with arm overlay."""
        with self._lock:
            if self._raw_frame is None:
                return None
            frame = self._raw_frame.copy()
            landmarks = self._latest_landmarks

        h, w = frame.shape[:2]

        # Draw arm landmarks if available
        if landmarks is not None:
            elbow_idx = self._arm_lm["elbow"]
            wrist_idx = self._arm_lm["wrist"]
            index_idx = self._arm_lm["index"]
            pinky_idx = self._arm_lm["pinky"]

            pts = {
                "elbow": (int(landmarks[elbow_idx][0] * w), int(landmarks[elbow_idx][1] * h)),
                "wrist": (int(landmarks[wrist_idx][0] * w), int(landmarks[wrist_idx][1] * h)),
                "index": (int(landmarks[index_idx][0] * w), int(landmarks[index_idx][1] * h)),
                "pinky": (int(landmarks[pinky_idx][0] * w), int(landmarks[pinky_idx][1] * h)),
            }
            hand_center = (
                (pts["index"][0] + pts["pinky"][0]) // 2,
                (pts["index"][1] + pts["pinky"][1]) // 2,
            )

            cv2.line(frame, pts["elbow"], pts["wrist"], (0, 255, 0), 3)
            cv2.line(frame, pts["wrist"], hand_center, (0, 255, 255), 3)
            cv2.line(frame, pts["index"], pts["pinky"], (100, 100, 0), 1)

            cv2.circle(frame, pts["elbow"], 8, (255, 0, 0), -1)
            cv2.circle(frame, pts["wrist"], 8, (0, 0, 255), -1)
            cv2.circle(frame, hand_center, 8, (0, 255, 255), -1)

            for name, pt in [("ELBOW", pts["elbow"]), ("WRIST", pts["wrist"]), ("MID", hand_center)]:
                cv2.putText(frame, name, (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, f"[POSE] {label.upper()}", (15, 30),
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

        filename = f"pose_stim{self._stim_count:03d}_{label}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _average_landmarks(self, landmarks_list: List[PoseLandmarks]) -> PoseLandmarks:
        """Average multiple frames of landmarks for stability."""
        arr = np.array(landmarks_list)
        avg = np.mean(arr, axis=0)
        return [tuple(row) for row in avg]

    def _ensure_model(self, path: Optional[str]) -> Optional[str]:
        """Download pose landmarker model if needed."""
        if path and os.path.exists(path):
            return path

        default_path = "pose_landmarker.task"
        if os.path.exists(default_path):
            return default_path

        print("  Downloading pose landmarker model (heavy)...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_heavy/float16/1/"
               "pose_landmarker_heavy.task")
        try:
            urllib.request.urlretrieve(url, default_path)
            print(f"  ✓ Model downloaded: {default_path}")
            return default_path
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return None

    def _print_result(self, r: WristMovementResult) -> None:
        """Print wrist movement report to terminal."""
        print()
        print("=" * 60)
        print("  WRIST MOVEMENT RESULT (Pose Tracker)")
        print("=" * 60)

        if r.channel is not None:
            print(f"  Stimulation: Ch{r.channel}, {r.intensity_ma}mA, {r.pulse_width_us}us")

        print(f"  Arm: {r.arm_side}")
        print(f"  Latency: {r.latency_ms:.0f} ms")
        print(f"  Visibility: {r.visibility:.0%}")
        print()
        print(f"  --- Wrist Angle ---")
        print(f"  Baseline: {r.baseline_wrist_angle:+.1f} deg")
        print(f"  Peak:     {r.result_wrist_angle:+.1f} deg")
        print(f"  Delta:    {r.wrist_angle_delta:+.1f} deg")
        print(f"  Result:   {r.wrist_feedback}")
        print("=" * 60)
        print()