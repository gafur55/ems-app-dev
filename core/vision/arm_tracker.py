"""
Arm tracker — Layer 2 of the new ArUco tracking system.

Detects the wrist ArUco marker (ID=1) and uses it together with the
manually entered arm length to establish the full forearm coordinate
system. Contour detection on the black mat finds the actual arm
boundary, giving per-point width measurements that account for each
person's individual forearm shape.

Coordinate system:
    s = 0.0  →  wrist (ArUco ID 1 center)
    s = 1.0  →  elbow (computed: wrist_px + arm_dir * arm_length_mm * px_per_mm)
    t = 0.0  →  arm centerline
    t = +1.0 →  thumb / radial side
    t = -1.0 →  pinky / ulnar side

All text is rendered by PyQt in the UI layer — draw_overlay() only draws geometry.
"""

import cv2
import numpy as np
from cv2 import aruco
from collections import deque
from typing import Optional, Dict, List, Tuple

from config import settings


# ── Constants ────────────────────────────────────────────────────────────────
WRIST_ID      = settings.WRIST_ID
SMOOTH_FRAMES = 10

LAB_BLUR_KERNEL = 5
MORPH_KERNEL    = 15
MIN_ARM_AREA_PX = 5000


class ArmTracker:
    """
    Tracks the forearm using:
      1. ArUco ID=1 on the wrist → anchor point (s=0)
      2. Arm length (manually entered) → elbow position (s=1)
      3. OpenCV contour detection on the arm silhouette → width profile
    """

    def __init__(self):
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self._detector   = aruco.ArucoDetector(
            self._aruco_dict, aruco.DetectorParameters()
        )

        self._wrist_buffer:  deque              = deque(maxlen=SMOOTH_FRAMES)
        self._arm_length_mm: Optional[float]    = None
        self._wrist_px:      Optional[np.ndarray] = None
        self._elbow_px:      Optional[np.ndarray] = None
        self._arm_dir:       Optional[np.ndarray] = None
        self._perp_dir:      Optional[np.ndarray] = None
        self._arm_contour:   Optional[np.ndarray] = None
        self._width_profile: List[dict]          = []
        self._ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_arm_length(self, arm_length_cm: float) -> None:
        self._arm_length_mm = arm_length_cm * 10.0
        print(f"  [ArmTracker] Arm length set: {arm_length_cm} cm "
              f"({self._arm_length_mm:.0f} mm)")

    def process(self, frame: np.ndarray, mat_state: dict) -> dict:
        """
        Run wrist detection and contour analysis on a frame.
        """
        if not mat_state.get("ready") or self._arm_length_mm is None:
            self._ready = False
            return {
                "ready":      False,
                "wrist_fill": len(self._wrist_buffer),
                "has_length": self._arm_length_mm is not None,
                "mat_ready":  mat_state.get("ready", False),
            }

        px_per_mm = mat_state["px_per_mm"]

        # Detect wrist ArUco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid == WRIST_ID:
                    center = self._marker_center(corners[i])
                    self._wrist_buffer.append(center)
                    break

        if len(self._wrist_buffer) == 0:
            self._ready = False
            return {"ready": False, "wrist_fill": 0,
                    "has_length": True, "mat_ready": True}

        wrist_px      = np.mean(self._wrist_buffer, axis=0).astype(np.float32)
        arm_length_px = self._arm_length_mm * px_per_mm
        arm_dir       = self._estimate_arm_direction(frame, wrist_px, arm_length_px)
        elbow_px      = wrist_px + arm_dir * arm_length_px
        arm_contour   = self._detect_arm_contour(frame)

        if arm_contour is not None:
            perp_dir      = np.array([-arm_dir[1], arm_dir[0]])
            width_profile = self._compute_width_profile(
                arm_contour, wrist_px, arm_dir, perp_dir,
                arm_length_px, px_per_mm
            )
        else:
            perp_dir      = np.array([-arm_dir[1], arm_dir[0]])
            width_profile = []

        H        = mat_state["homography"]
        wrist_mm = self._px_to_mm(wrist_px, H)
        elbow_mm = self._px_to_mm(elbow_px, H)

        self._wrist_px      = wrist_px
        self._elbow_px      = elbow_px
        self._arm_dir       = arm_dir
        self._perp_dir      = perp_dir
        self._arm_contour   = arm_contour
        self._width_profile = width_profile
        self._ready         = True

        return {
            "ready":           True,
            "wrist_center_px": wrist_px,
            "wrist_center_mm": wrist_mm,
            "elbow_center_px": elbow_px,
            "elbow_center_mm": elbow_mm,
            "arm_dir":         arm_dir,
            "perp_dir":        perp_dir,
            "arm_length_mm":   self._arm_length_mm,
            "arm_length_px":   arm_length_px,
            "arm_contour":     arm_contour,
            "width_profile":   width_profile,
            "wrist_fill":      len(self._wrist_buffer),
        }

    @property
    def is_ready(self) -> bool:
        return self._ready

    def draw_overlay(self, frame: np.ndarray, arm_state: dict) -> np.ndarray:
        """
        Draw arm tracking overlay — geometry only, no text.
        All text is rendered by PyQt in the UI layer.
        """
        if not arm_state.get("ready"):
            return frame

        wrist_px = arm_state["wrist_center_px"].astype(int)
        elbow_px = arm_state["elbow_center_px"].astype(int)

        # Arm contour
        contour = arm_state.get("arm_contour")
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 180, 255), 2)

        # Arm axis line wrist → elbow
        cv2.line(frame, tuple(wrist_px), tuple(elbow_px),
                 (0, 255, 100), 2)

        # Width profile sample dots every 20%
        for wp in arm_state.get("width_profile", []):
            if wp["s"] % 0.2 < 0.05:
                cx, cy = int(wp["center_px"][0]), int(wp["center_px"][1])
                cv2.circle(frame, (cx, cy), 3, (255, 200, 0), -1)

        # Wrist dot (cyan)
        cv2.circle(frame, tuple(wrist_px), 10, (0, 255, 255), -1)
        cv2.circle(frame, tuple(wrist_px), 10, (255, 255, 255), 2)

        # Elbow dot (orange)
        cv2.circle(frame, tuple(elbow_px), 10, (255, 100, 0), -1)
        cv2.circle(frame, tuple(elbow_px), 10, (255, 255, 255), 2)

        return frame

    # ── Contour detection ─────────────────────────────────────────────────────

    def _detect_arm_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect arm silhouette using LAB a* channel against dark mat."""
        lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        a_ch     = lab[:, :, 1]
        blurred  = cv2.GaussianBlur(a_ch, (LAB_BLUR_KERNEL, LAB_BLUR_KERNEL), 0)
        _, mask  = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL)
        )
        mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        valid = [c for c in contours if cv2.contourArea(c) > MIN_ARM_AREA_PX]
        if not valid:
            return None
        return max(valid, key=cv2.contourArea)

    # ── Arm direction ─────────────────────────────────────────────────────────

    def _estimate_arm_direction(
        self,
        frame: np.ndarray,
        wrist_px: np.ndarray,
        arm_length_px: float,
    ) -> np.ndarray:
        """Estimate arm direction via PCA on contour, fallback to downward."""
        contour = self._detect_arm_contour(frame)
        if contour is not None and len(contour) > 5:
            pts = contour.reshape(-1, 2).astype(np.float32)
            _, eigenvectors = cv2.PCACompute(pts, mean=None)
            principal = eigenvectors[0]
            if principal[1] < 0:
                principal = -principal
            return principal.astype(np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    # ── Width profile ─────────────────────────────────────────────────────────

    def _compute_width_profile(
        self,
        contour: np.ndarray,
        wrist_px: np.ndarray,
        arm_dir: np.ndarray,
        perp_dir: np.ndarray,
        arm_length_px: float,
        px_per_mm: float,
        n_samples: int = 20,
    ) -> List[dict]:
        """Sample arm width at N points along the arm axis."""
        profile = []
        pts     = contour.reshape(-1, 2).astype(np.float32)

        for i in range(n_samples):
            s         = i / (n_samples - 1)
            center_px = wrist_px + arm_dir * (s * arm_length_px)
            relative  = pts - center_px
            along     = np.dot(relative, arm_dir)
            lateral   = np.dot(relative, perp_dir)
            tolerance = arm_length_px * 0.05
            nearby    = lateral[np.abs(along) < tolerance]

            if len(nearby) < 2:
                continue

            width_px = float(nearby.max() - nearby.min())
            width_mm = width_px / px_per_mm

            profile.append({
                "s":         s,
                "width_mm":  width_mm,
                "width_px":  width_px,
                "center_px": center_px,
            })

        return profile

    def get_width_at_s(self, s: float) -> Optional[float]:
        """Get interpolated arm width in mm at a given s position."""
        if not self._width_profile:
            return None
        s_vals = [wp["s"] for wp in self._width_profile]
        w_vals = [wp["width_mm"] for wp in self._width_profile]
        if len(s_vals) < 2:
            return w_vals[0] if w_vals else None
        return float(np.interp(s, s_vals, w_vals))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _marker_center(corners) -> np.ndarray:
        c = corners[0]
        return np.array([c[:, 0].mean(), c[:, 1].mean()], dtype=np.float32)

    @staticmethod
    def _px_to_mm(px_point: np.ndarray, H: np.ndarray) -> Tuple[float, float]:
        pt     = np.array([[[px_point[0], px_point[1]]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)
        return float(result[0][0][0]), float(result[0][0][1])