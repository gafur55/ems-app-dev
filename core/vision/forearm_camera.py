"""
Threaded camera reader for the live forearm view.

Integrates ArucoTracker — every frame is passed through ArUco detection
before being stored. The annotated frame (with overlays) is what gets
pushed to the UI. ArUco state is available via get_aruco_state().

Usage:
    camera = ForearmCamera(camera_index=2)
    camera.start()

    # Get annotated live frame for display
    frame = camera.get_frame()

    # Get ArUco measurement state (thread-safe copy)
    state = camera.get_aruco_state()
    if state["ready"]:
        print(state["forearm_length_mm"])
        for eid, info in state["electrodes"].items():
            print(eid, info["down_mm"], info["lateral_mm"])

    camera.stop()
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional

from core.vision.undistorter import Undistorter
from core.vision.aruco_tracker import ArucoTracker


class ForearmCamera:
    """
    Threaded camera reader for live forearm display with built-in ArUco tracking.

    Reads frames continuously in a background thread, applies undistortion,
    runs ArUco detection, and stores the annotated frame + ArUco state.
    """

    # Tilt correction — adjust until live feed looks straight
    TILT_ANGLE_DEG = -15

    def __init__(self, camera_index: int = 2):
        self.camera_index = camera_index
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._lock       = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._aruco_state:  dict = {"ready": False}

        self._undistorter: Optional[Undistorter] = None
        self._aruco_tracker: Optional[ArucoTracker] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start camera reader + ArUco tracker."""
        self._undistorter  = Undistorter(self.camera_index)
        self._aruco_tracker = ArucoTracker()

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

        print(f"✓ Forearm camera started (index {self.camera_index}) — waiting for frames...")
        return True

    def stop(self) -> None:
        """Stop the camera reader thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._latest_frame = None
            self._aruco_state  = {"ready": False}
        print("✓ Forearm camera stopped")

    @property
    def is_running(self) -> bool:
        return self._running

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
        Get the latest ArUco detection state.

        Returns a dict with at minimum {"ready": bool}.
        When ready=True, also contains:
            px_per_mm, wrist_width_mm, elbow_width_mm, forearm_length_mm,
            wrist_center_px, elbow_center_px, origin_px,
            wrist_pinky_px, wrist_thumb_px, elbow_pinky_px, elbow_thumb_px,
            down_axis, lateral_axis,
            electrodes: {id: {pixel, down_mm, lateral_mm, side, buffer_full}}
        """
        with self._lock:
            return dict(self._aruco_state)

    def is_aruco_ready(self) -> bool:
        """Quick check: are all 4 reference markers tracked?"""
        with self._lock:
            return self._aruco_state.get("ready", False)

    # ── Camera loop ───────────────────────────────────────────────────────────

    def _camera_loop(self) -> None:
        """Background thread: read → undistort → ArUco → store."""
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

            # 1. Undistort
            frame = self._undistorter.apply(frame)

            # # 2. Optional tilt correction
            # if self.TILT_ANGLE_DEG != 0:
            #     frame = self._rotate(frame, self.TILT_ANGLE_DEG)

            # 3. ArUco detection + annotation
            annotated, state = self._aruco_tracker.process(frame)

            # 4. Store results
            with self._lock:
                self._latest_frame = annotated
                self._aruco_state  = state

        cap.release()
        print("  [ForearmCamera] Camera released")

    @staticmethod
    def _rotate(frame: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = frame.shape[:2]
        M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        return cv2.warpAffine(frame, M, (w, h))