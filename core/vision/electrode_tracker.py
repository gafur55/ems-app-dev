"""
Electrode tracker — Layer 3 of the new ArUco tracking system.

Detects electrode ArUco markers (IDs 5, 6, ...) and converts their
pixel positions to normalized anatomical coordinates (s, t) using
the arm geometry established by ArmTracker.

Also handles the REVERSE mapping: given (s, t) from a previous session,
compute the pixel position on the current person's arm where the
electrode should be placed — the "reproduce on new person" feature.

Coordinate system (from ArmTracker):
    s = 0.0 → wrist,   s = 1.0 → elbow
    t = 0.0 → centerline
    t > 0   → thumb/radial side
    t < 0   → pinky/ulnar side
    t is normalized by arm width at that s level

All text is rendered by PyQt in the UI layer — draw_overlay() only draws geometry.
"""

import cv2
import numpy as np
from cv2 import aruco
from collections import deque
from typing import Optional, List, Tuple, Dict

from config import settings


# ── Constants ────────────────────────────────────────────────────────────────
ELECTRODE_IDS = settings.ELECTRODE_IDS
SMOOTH_FRAMES = 8


class ElectrodeTracker:
    """
    Detects electrode ArUco markers and maps them to anatomical (s, t)
    coordinates using the arm geometry from ArmTracker.

    Also provides reverse mapping: (s, t) → pixel for placement suggestions.
    """

    def __init__(self, electrode_ids: Optional[List[int]] = None):
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self._detector   = aruco.ArucoDetector(
            self._aruco_dict, aruco.DetectorParameters()
        )

        self._electrode_ids = electrode_ids or ELECTRODE_IDS

        self._buffers: Dict[int, deque] = {
            eid: deque(maxlen=SMOOTH_FRAMES)
            for eid in self._electrode_ids
        }

        self._electrodes:  Dict[int, dict] = {}
        self._suggestions: List[dict]      = []

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, arm_state: dict) -> dict:
        """
        Detect electrode markers and compute their anatomical coordinates.
        """
        if not arm_state.get("ready"):
            return {"electrodes": {}, "suggestions": self._suggestions}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid in self._buffers:
                    center = self._marker_center(corners[i])
                    self._buffers[mid].append(center)

        electrodes = {}
        H          = arm_state.get("homography")

        for eid in self._electrode_ids:
            buf = self._buffers[eid]
            if not buf:
                continue

            px_pt            = np.mean(buf, axis=0).astype(np.float32)
            s, t, width_at_s = self._pixel_to_st(px_pt, arm_state)

            mm_pt = None
            if H is not None:
                pt     = np.array([[[px_pt[0], px_pt[1]]]], dtype=np.float32)
                result = cv2.perspectiveTransform(pt, H)
                mm_pt  = (float(result[0][0][0]), float(result[0][0][1]))

            electrodes[eid] = {
                "pixel":         px_pt,
                "mm":            mm_pt,
                "s":             round(s, 4),
                "t":             round(t, 4),
                "width_at_s_mm": round(width_at_s, 1) if width_at_s else None,
                "buffer_full":   len(buf) == SMOOTH_FRAMES,
                "side":          "thumb" if t > 0 else "pinky" if t < 0 else "center",
            }

        self._electrodes = electrodes
        resolved         = self._resolve_suggestions(arm_state)

        return {
            "electrodes":  electrodes,
            "suggestions": resolved,
        }

    def set_suggestions(self, placements: List[Tuple[float, float]],
                        labels: Optional[List[str]] = None) -> None:
        """Set electrode placement suggestions from a previous session."""
        self._suggestions = []
        for i, (s, t) in enumerate(placements):
            label = labels[i] if labels and i < len(labels) else f"E{i+1}"
            self._suggestions.append({"s": s, "t": t, "label": label})
        print(f"  [ElectrodeTracker] {len(self._suggestions)} suggestion(s) loaded")

    def clear_suggestions(self) -> None:
        self._suggestions = []

    def get_electrode_positions(self) -> Dict[int, dict]:
        return self._electrodes

    def draw_overlay(self, frame: np.ndarray, electrode_state: dict) -> np.ndarray:
        """
        Draw electrode markers and suggestions — geometry only, no text.
        All labels are rendered by PyQt in the UI layer.
        """
        electrode_colors = [
            (0, 255, 0),      # E1 green
            (0, 200, 255),    # E2 cyan
            (255, 100, 0),    # E3 orange
            (200, 0, 255),    # E4 purple
        ]

        # Detected electrodes — colored dots only
        for i, (eid, info) in enumerate(electrode_state["electrodes"].items()):
            px        = info["pixel"].astype(int)
            color     = electrode_colors[i % len(electrode_colors)]
            thickness = -1 if info["buffer_full"] else 2
            cv2.circle(frame, tuple(px), 12, color, thickness)
            cv2.circle(frame, tuple(px), 12, (255, 255, 255), 1)

        # Suggestions — crosshair dots only
        for suggestion in electrode_state.get("suggestions", []):
            px = suggestion.get("pixel")
            if px is None:
                continue
            px = (int(px[0]), int(px[1]))
            cv2.circle(frame, px, 16, (255, 255, 0), 2)
            cv2.circle(frame, px, 4,  (255, 255, 0), -1)
            cv2.line(frame, (px[0] - 20, px[1]), (px[0] + 20, px[1]),
                     (255, 255, 0), 1)
            cv2.line(frame, (px[0], px[1] - 20), (px[0], px[1] + 20),
                     (255, 255, 0), 1)

        return frame

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def _pixel_to_st(
        self,
        px_pt: np.ndarray,
        arm_state: dict,
    ) -> Tuple[float, float, Optional[float]]:
        """Convert pixel position to normalized (s, t) coordinates."""
        wrist_px      = arm_state["wrist_center_px"]
        arm_dir       = arm_state["arm_dir"]
        perp_dir      = arm_state["perp_dir"]
        arm_length_px = arm_state["arm_length_px"]
        width_profile = arm_state.get("width_profile", [])
        px_per_mm     = arm_state.get("px_per_mm", 1.0)

        relative   = px_pt - wrist_px
        along_px   = float(np.dot(relative, arm_dir))
        lateral_px = float(np.dot(relative, perp_dir))
        s          = along_px / arm_length_px

        width_at_s_mm = self._interpolate_width(s, width_profile)
        if width_at_s_mm and width_at_s_mm > 0:
            t = lateral_px / (width_at_s_mm * px_per_mm / 2.0)
        else:
            t = lateral_px / (arm_length_px * 0.15)

        return s, t, width_at_s_mm

    def st_to_pixel(
        self,
        s: float,
        t: float,
        arm_state: dict,
    ) -> Optional[np.ndarray]:
        """Convert normalized (s, t) to pixel coordinates on the current person."""
        if not arm_state.get("ready"):
            return None

        wrist_px      = arm_state["wrist_center_px"]
        arm_dir       = arm_state["arm_dir"]
        perp_dir      = arm_state["perp_dir"]
        arm_length_px = arm_state["arm_length_px"]
        width_profile = arm_state.get("width_profile", [])
        px_per_mm     = arm_state.get("px_per_mm", 1.0)

        along_px      = s * arm_length_px
        width_at_s_mm = self._interpolate_width(s, width_profile)
        if width_at_s_mm and width_at_s_mm > 0:
            lateral_px = t * (width_at_s_mm * px_per_mm / 2.0)
        else:
            lateral_px = t * (arm_length_px * 0.15)

        pixel = wrist_px + arm_dir * along_px + perp_dir * lateral_px
        return pixel.astype(np.float32)

    def _resolve_suggestions(self, arm_state: dict) -> List[dict]:
        resolved = []
        for suggestion in self._suggestions:
            px = self.st_to_pixel(suggestion["s"], suggestion["t"], arm_state)
            resolved.append({**suggestion, "pixel": px})
        return resolved

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _marker_center(corners) -> np.ndarray:
        c = corners[0]
        return np.array([c[:, 0].mean(), c[:, 1].mean()], dtype=np.float32)

    @staticmethod
    def _interpolate_width(s: float, width_profile: List[dict]) -> Optional[float]:
        if not width_profile:
            return None
        s_vals = [wp["s"] for wp in width_profile]
        w_vals = [wp["width_mm"] for wp in width_profile]
        if len(s_vals) < 2:
            return w_vals[0] if w_vals else None
        return float(np.interp(s, s_vals, w_vals))

    @staticmethod
    def _get_homography_from_mat() -> Optional[np.ndarray]:
        return None