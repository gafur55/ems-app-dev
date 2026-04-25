"""
Threaded camera reader for the live forearm view.

Orchestrates the 3-layer tracking system:
    Layer 1: MatCalibrator  — 4 corner ArUcos → homography + px/mm
    Layer 2: ArmTracker     — wrist ArUco + arm length + contours → anatomy
    Layer 3: ElectrodeTracker — electrode ArUcos → (s,t) + suggestions

The annotated frame (with all overlays) is pushed to the UI via QTimer.
The full state dict is available via get_aruco_state() — same interface
as the old ForearmCamera so main_window.py needs minimal changes.

Usage:
    camera = ForearmCamera(camera_index=2)
    camera.set_arm_length(27.5)           # from sign-in
    camera.start()

    frame = camera.get_frame()            # latest annotated frame
    state = camera.get_aruco_state()      # full tracking state

    # Load suggestions from previous session:
    camera.set_electrode_suggestions([(0.4, 0.1), (0.35, -0.05)])

    camera.stop()
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, List, Tuple

from core.vision.mat_calibrator  import MatCalibrator
from core.vision.arm_tracker     import ArmTracker
from core.vision.electrode_tracker import ElectrodeTracker

from core.vision.mat_calibrator    import MatCalibrator
from core.vision.arm_tracker       import ArmTracker
from core.vision.electrode_tracker import ElectrodeTracker
from core.vision.undistorter       import Undistorter   # NEW


class ForearmCamera:
    """
    Threaded camera reader with 3-layer ArUco tracking.

    Reads frames continuously in a background thread, runs all three
    tracking layers in sequence, and stores the annotated frame + state.

    Public interface is unchanged from the previous ForearmCamera —
    get_frame() and get_aruco_state() work the same way.
    """

    def __init__(self, camera_index: int = 2):
        self.camera_index = camera_index

        self._undistorter = Undistorter(camera_index)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock   = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._state: dict = {"ready": False, "mat_ready": False, "homography": None}

        # The 3 tracking layers
        self._mat_calibrator    = MatCalibrator()
        self._arm_tracker       = ArmTracker()
        self._electrode_tracker = ElectrodeTracker()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start camera reader thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        time.sleep(0.5)

        with self._lock:
            if self._latest_frame is not None:
                print(f"✓ Forearm camera started (index {self.camera_index})")
                return True

        if not self._running:
            print(f"✗ Forearm camera failed (index {self.camera_index})")
            return False

        print(f"✓ Forearm camera started (index {self.camera_index}) "
              f"— waiting for frames...")
        return True

    def stop(self) -> None:
        """Stop the camera reader thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._latest_frame = None
            self._state = {"ready": False}
        print("✓ Forearm camera stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_arm_length(self, arm_length_cm: float) -> None:
        """
        Set the participant's forearm length.
        Call this after sign-in before the session starts.

        Args:
            arm_length_cm: Wrist-to-elbow length in cm from sign-in data.
        """
        self._arm_tracker.set_arm_length(arm_length_cm)

    def set_electrode_suggestions(
        self,
        placements: List[Tuple[float, float]],
        labels: Optional[List[str]] = None,
    ) -> None:
        """
        Load electrode placement suggestions from a previous session.
        These will be overlaid as target dots on the live feed.

        Args:
            placements: List of (s, t) tuples from previous session DB.
            labels:     Optional channel labels (e.g. ["Ch0-A", "Ch0-B"]).
        """
        self._electrode_tracker.set_suggestions(placements, labels)

    def clear_suggestions(self) -> None:
        """Remove all placement suggestions from the overlay."""
        self._electrode_tracker.clear_suggestions()

    # ── Data access ───────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest annotated camera frame (BGR).
        Returns None if no frame is available yet.
        """
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def get_aruco_state(self) -> dict:
        """
        Get the latest full tracking state.

        Returns a dict with at minimum {"ready": bool}.
        When ready=True (all layers ready) also contains:
            px_per_mm, homography
            wrist_center_px, wrist_center_mm
            elbow_center_px, elbow_center_mm
            forearm_length_mm
            arm_contour, width_profile
            electrodes: {id: {pixel, mm, s, t, side, buffer_full}}
            suggestions: [{s, t, pixel, label}]
        """
        with self._lock:
            return dict(self._state)

    def is_aruco_ready(self) -> bool:
        """Quick check: mat calibrated + wrist found + arm length set."""
        with self._lock:
            return self._state.get("ready", False)

    def is_mat_ready(self) -> bool:
        """Check if mat calibration alone is ready."""
        with self._lock:
            return self._state.get("mat_ready", False)

    # ── Camera loop ───────────────────────────────────────────────────────────

    def _camera_loop(self) -> None:
        """Background thread: read → detect → annotate → store."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"✗ [ForearmCamera] Could not open camera {self.camera_index}")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        print(f"  [ForearmCamera] Camera {self.camera_index} opened")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = self._undistorter.apply(frame)
            
            annotated, state = self._process_frame(frame)

            with self._lock:
                self._latest_frame = annotated
                self._state        = state

        cap.release()
        print("  [ForearmCamera] Camera released")

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Run all 3 tracking layers on a single frame and build the combined
        state dict.

        Args:
            frame: Raw BGR frame from camera.

        Returns:
            (annotated_frame, state_dict)
        """
        annotated = frame.copy()

        # ── Layer 1: Mat calibration ──────────────────────────────────────────
        mat_state = self._mat_calibrator.process(frame)
        annotated = self._mat_calibrator.draw_overlay(annotated)

        if not mat_state["ready"]:
            state = {
                "ready":     False,
                "mat_ready": False,
                "fills":     mat_state.get("fills", {}),
            }
            return annotated, state

        # ── Layer 2: Arm tracking ─────────────────────────────────────────────
        arm_state = self._arm_tracker.process(frame, mat_state)
        # Pass homography through for electrode tracker
        arm_state["homography"] = mat_state["homography"]
        arm_state["px_per_mm"]  = mat_state["px_per_mm"]

        annotated = self._arm_tracker.draw_overlay(annotated, arm_state)

        if not arm_state["ready"]:
            state = {
                "ready":      False,
                "mat_ready":  True,
                "arm_ready":  False,
                "wrist_fill": arm_state.get("wrist_fill", 0),
                "has_length": arm_state.get("has_length", False),
                "px_per_mm":  mat_state["px_per_mm"],
                "homography": mat_state["homography"],
            }
            return annotated, state

        # ── Layer 3: Electrode tracking ───────────────────────────────────────
        electrode_state = self._electrode_tracker.process(frame, arm_state)
        annotated = self._electrode_tracker.draw_overlay(annotated, electrode_state)

        # ── Build combined state (same shape as old aruco_tracker state) ──────
        state = {
            "ready":             True,
            "mat_ready":         True,
            "arm_ready":         True,

            # Scale
            "px_per_mm":         mat_state["px_per_mm"],
            "homography":        mat_state["homography"],

            # Arm anatomy
            "wrist_center_px":   arm_state["wrist_center_px"],
            "wrist_center_mm":   arm_state["wrist_center_mm"],
            "elbow_center_px":   arm_state["elbow_center_px"],
            "elbow_center_mm":   arm_state["elbow_center_mm"],
            "forearm_length_mm": arm_state["arm_length_mm"],
            "arm_dir":           arm_state["arm_dir"],
            "perp_dir":          arm_state["perp_dir"],
            "arm_contour":       arm_state.get("arm_contour"),
            "width_profile":     arm_state.get("width_profile", []),

            # Electrodes
            "electrodes":        electrode_state["electrodes"],
            "suggestions":       electrode_state["suggestions"],

            # Compat fields for CalibrationManager / AnatomicalCoordinateMapper
            "wrist_pinky_px":    arm_state["wrist_center_px"],  # approx
            "wrist_thumb_px":    arm_state["wrist_center_px"],  # refined by contour
            "elbow_pinky_px":    arm_state["elbow_center_px"],
            "elbow_thumb_px":    arm_state["elbow_center_px"],
            "wrist_width_mm":    self._arm_tracker.get_width_at_s(0.0) or 0.0,
            "elbow_width_mm":    self._arm_tracker.get_width_at_s(1.0) or 0.0,
        }

        return annotated, state
