"""
Arm tracker — Layer 2 of the ArUco tracking system.

Three-marker approach:
    ArUco ID = WRIST_ID         on the WRIST   → s = 0 anchor (longitudinal origin)
    ArUco ID = ELBOW_ID         on the ELBOW   → defines arm DIRECTION (toward elbow)
    ArUco ID = WRIST_RADIAL_ID  on the WRIST,
               thumb/radial side               → defines lateral DIRECTION (+t = thumb)

Arm length still comes from the manually-entered DB value — the elbow marker
only defines the *direction*, then we extrapolate `arm_length_mm` along that
direction from the wrist to get the s=1 point.

The radial marker fixes the long-standing problem that the lateral axis was
just a 90° rotation of the arm axis in image space, which was only correct
for a perfectly top-down camera and a flat arm. With a physical radial
marker, the lateral axis tracks the true anatomical thumb side regardless
of camera angle or wrist rotation.

Coordinate system:
    s = 0.0  →  wrist (wrist marker center)
    s = 1.0  →  elbow point (= wrist + arm_dir × arm_length_mm)
    t = 0.0  →  arm centerline
    t = +1.0 →  thumb / radial side
    t = -1.0 →  pinky / ulnar side
"""

import cv2
import numpy as np
from cv2 import aruco
from collections import deque
from typing import Optional, Dict, List, Tuple

from config import settings


# ── Constants ────────────────────────────────────────────────────────────────
WRIST_ID        = settings.WRIST_ID
ELBOW_ID        = settings.ELBOW_ID
WRIST_RADIAL_ID = settings.WRIST_RADIAL_ID
SMOOTH_FRAMES   = 10

LAB_BLUR_KERNEL = 5
MORPH_KERNEL    = 15
MIN_ARM_AREA_PX = 5000


class ArmTracker:
    """
    Tracks the forearm using:
      1. Wrist ArUco         → s = 0 anchor
      2. Elbow ArUco         → defines arm direction (wrist → elbow)
      3. Wrist radial ArUco  → defines lateral direction (thumb side = +t)
      4. Arm length from DB  → defines the actual elbow point
                               (= wrist + arm_dir × arm_length_mm)
      5. OpenCV contour      → forearm width profile (for t normalization)
    """

    def __init__(self):
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self._detector   = aruco.ArucoDetector(
            self._aruco_dict, aruco.DetectorParameters()
        )

        self._wrist_buffer:        deque = deque(maxlen=SMOOTH_FRAMES)
        self._elbow_buffer:        deque = deque(maxlen=SMOOTH_FRAMES)
        self._wrist_radial_buffer: deque = deque(maxlen=SMOOTH_FRAMES)

        self._arm_length_mm:   Optional[float]      = None
        self._wrist_px:        Optional[np.ndarray] = None
        self._elbow_px:        Optional[np.ndarray] = None
        self._wrist_radial_px: Optional[np.ndarray] = None
        self._arm_dir:         Optional[np.ndarray] = None
        self._perp_dir:        Optional[np.ndarray] = None
        self._arm_contour:     Optional[np.ndarray] = None
        self._width_profile:   List[dict]           = []
        self._ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_arm_length(self, arm_length_cm: float) -> None:
        self._arm_length_mm = arm_length_cm * 10.0
        print(f"  [ArmTracker] Arm length set: {arm_length_cm} cm "
              f"({self._arm_length_mm:.0f} mm)")

    def process(self, frame: np.ndarray, mat_state: dict) -> dict:
        """Run wrist + elbow + radial detection + contour analysis."""
        if not mat_state.get("ready") or self._arm_length_mm is None:
            self._ready = False
            return {
                "ready":             False,
                "wrist_fill":        len(self._wrist_buffer),
                "elbow_fill":        len(self._elbow_buffer),
                "wrist_radial_fill": len(self._wrist_radial_buffer),
                "has_length":        self._arm_length_mm is not None,
                "mat_ready":         mat_state.get("ready", False),
                "missing":           self._missing_markers(),
            }

        px_per_mm = mat_state["px_per_mm"]

        # ── Detect all three markers ──────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid == WRIST_ID:
                    self._wrist_buffer.append(self._marker_center(corners[i]))
                elif mid == ELBOW_ID:
                    self._elbow_buffer.append(self._marker_center(corners[i]))
                elif mid == WRIST_RADIAL_ID:
                    self._wrist_radial_buffer.append(self._marker_center(corners[i]))

        # ── All three markers required ────────────────────────────────────────
        if (len(self._wrist_buffer) == 0
                or len(self._elbow_buffer) == 0
                or len(self._wrist_radial_buffer) == 0):
            self._ready = False
            return {
                "ready":             False,
                "wrist_fill":        len(self._wrist_buffer),
                "elbow_fill":        len(self._elbow_buffer),
                "wrist_radial_fill": len(self._wrist_radial_buffer),
                "has_length":        True,
                "mat_ready":         True,
                "missing":           self._missing_markers(),
            }

        # ── Compute geometry ──────────────────────────────────────────────────
        wrist_px        = np.mean(self._wrist_buffer,        axis=0).astype(np.float32)
        elbow_marker_px = np.mean(self._elbow_buffer,        axis=0).astype(np.float32)
        wrist_radial_px = np.mean(self._wrist_radial_buffer, axis=0).astype(np.float32)

        # Longitudinal direction: wrist → elbow
        diff = elbow_marker_px - wrist_px
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm < 1e-3:
            self._ready = False
            return {
                "ready":             False,
                "wrist_fill":        len(self._wrist_buffer),
                "elbow_fill":        len(self._elbow_buffer),
                "wrist_radial_fill": len(self._wrist_radial_buffer),
                "has_length":        True,
                "mat_ready":         True,
                "missing":           [],
                "error":             "wrist and elbow markers overlap",
            }
        arm_dir = (diff / diff_norm).astype(np.float32)

        # Lateral direction from the physical radial marker.
        # Thumb side = positive. Orthogonalize against arm_dir so the two axes
        # are guaranteed perpendicular even if the marker isn't placed perfectly
        # on the line perpendicular to the arm at the wrist.
        lat_raw  = wrist_radial_px - wrist_px
        lat_raw  = lat_raw - np.dot(lat_raw, arm_dir) * arm_dir
        lat_norm = float(np.linalg.norm(lat_raw))
        if lat_norm < 1e-3:
            self._ready = False
            return {
                "ready":             False,
                "wrist_fill":        len(self._wrist_buffer),
                "elbow_fill":        len(self._elbow_buffer),
                "wrist_radial_fill": len(self._wrist_radial_buffer),
                "has_length":        True,
                "mat_ready":         True,
                "missing":           [],
                "error":             "radial marker lies on the arm axis",
            }
        perp_dir = (lat_raw / lat_norm).astype(np.float32)

        # The "elbow point" used for s=1 — extrapolated from the DB arm length
        # along the direction defined by the two markers.
        arm_length_px = self._arm_length_mm * px_per_mm
        elbow_px      = wrist_px + arm_dir * arm_length_px

        # Contour & width profile (still useful for t-normalization)
        arm_contour = self._detect_arm_contour(frame)
        if arm_contour is not None:
            width_profile = self._compute_width_profile(
                arm_contour, wrist_px, arm_dir, perp_dir,
                arm_length_px, px_per_mm,
            )
        else:
            width_profile = []

        # Mat-frame mm coords (for the EPM / DB)
        H        = mat_state["homography"]
        wrist_mm = self._px_to_mm(wrist_px, H)
        elbow_mm = self._px_to_mm(elbow_px, H)

        # Cache & return
        self._wrist_px        = wrist_px
        self._elbow_px        = elbow_px
        self._wrist_radial_px = wrist_radial_px
        self._arm_dir         = arm_dir
        self._perp_dir        = perp_dir
        self._arm_contour     = arm_contour
        self._width_profile   = width_profile
        self._ready           = True

        return {
            "ready":             True,
            "wrist_center_px":   wrist_px,
            "wrist_center_mm":   wrist_mm,
            "elbow_center_px":   elbow_px,
            "elbow_center_mm":   elbow_mm,
            "elbow_marker_px":   elbow_marker_px,   # raw marker position (for overlay)
            "wrist_radial_px":   wrist_radial_px,   # raw radial marker (for overlay/debug)
            "arm_dir":           arm_dir,
            "perp_dir":          perp_dir,
            "arm_length_mm":     self._arm_length_mm,
            "arm_length_px":     arm_length_px,
            "arm_contour":       arm_contour,
            "width_profile":     width_profile,
            "wrist_fill":        len(self._wrist_buffer),
            "elbow_fill":        len(self._elbow_buffer),
            "wrist_radial_fill": len(self._wrist_radial_buffer),
        }

    @property
    def is_ready(self) -> bool:
        return self._ready

    def draw_overlay(self, frame: np.ndarray, arm_state: dict) -> np.ndarray:
        if not arm_state.get("ready"):
            return frame

        wrist_px        = arm_state["wrist_center_px"].astype(int)
        elbow_px        = arm_state["elbow_center_px"].astype(int)
        elbow_marker_px = arm_state.get("elbow_marker_px")
        wrist_radial_px = arm_state.get("wrist_radial_px")

        # Arm contour (background context)
        contour = arm_state.get("arm_contour")
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 180, 255), 2)

        # Wrist → elbow marker line (green)
        if elbow_marker_px is not None:
            em = elbow_marker_px.astype(int)
            cv2.line(frame, tuple(wrist_px), tuple(em),
                     (0, 255, 100), 2)

        # Wrist → radial marker line (lime) — shows the lateral axis
        if wrist_radial_px is not None:
            wr = wrist_radial_px.astype(int)
            cv2.line(frame, tuple(wrist_px), tuple(wr),
                     (100, 255, 100), 2)

        # Width profile sample dots every 20%
        for wp in arm_state.get("width_profile", []):
            if wp["s"] % 0.2 < 0.05:
                cx, cy = int(wp["center_px"][0]), int(wp["center_px"][1])
                cv2.circle(frame, (cx, cy), 3, (255, 200, 0), -1)

        # Wrist (cyan)
        cv2.circle(frame, tuple(wrist_px), 10, (0, 255, 255), -1)
        cv2.circle(frame, tuple(wrist_px), 10, (255, 255, 255), 2)

        # Elbow point at s=1 (orange) — DB length extrapolation
        cv2.circle(frame, tuple(elbow_px), 10, (255, 100, 0), -1)
        cv2.circle(frame, tuple(elbow_px), 10, (255, 255, 255), 2)

        # Raw elbow marker position (small magenta dot, distinct from s=1)
        if elbow_marker_px is not None:
            ep = elbow_marker_px.astype(int)
            cv2.circle(frame, tuple(ep), 6, (255, 0, 200), -1)
            cv2.circle(frame, tuple(ep), 6, (255, 255, 255), 1)

        # Raw radial marker position (small lime dot)
        if wrist_radial_px is not None:
            wr = wrist_radial_px.astype(int)
            cv2.circle(frame, tuple(wr), 6, (100, 255, 100), -1)
            cv2.circle(frame, tuple(wr), 6, (255, 255, 255), 1)

        return frame

    # ── Helpers for "missing markers" status ──────────────────────────────────

    def _missing_markers(self) -> List[str]:
        missing = []
        if len(self._wrist_buffer) == 0:
            missing.append("wrist")
        if len(self._elbow_buffer) == 0:
            missing.append("elbow")
        if len(self._wrist_radial_buffer) == 0:
            missing.append("wrist_radial")
        return missing

    # ── Contour detection ─────────────────────────────────────────────────────

    def _detect_arm_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        a_ch    = lab[:, :, 1]
        blurred = cv2.GaussianBlur(a_ch, (LAB_BLUR_KERNEL, LAB_BLUR_KERNEL), 0)
        _, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        valid = [c for c in contours if cv2.contourArea(c) > MIN_ARM_AREA_PX]
        if not valid:
            return None
        return max(valid, key=cv2.contourArea)

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