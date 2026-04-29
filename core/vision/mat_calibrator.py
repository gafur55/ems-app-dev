"""
Mat calibrator — Layer 1 of the new ArUco tracking system.

Detects 4 corner ArUco markers fixed to the mat corners and computes
a homography matrix H that maps any camera pixel to a real-world mm
coordinate. This corrects perspective and camera angle in one shot.

Mat layout (top-down camera, arm rests in the middle):

   ID=0 ──────────────── ID=2
  │                     │
  │    arm rests here   │
  │                     │
    ID=4 ──────────────── ID=3

Physical dimensions (set in settings.py):
    MAT_WIDTH_MM  = 165   (ID4 → ID0, and ID3 → ID2)
    MAT_HEIGHT_MM = 400   (ID4 → ID3, and ID0 → ID2)

Real-world coordinate origin = ID4 (top-left corner of mat).
    x increases → (toward ID0)
    y increases ↓ (toward ID3)

Usage:
    calibrator = MatCalibrator()
    result = calibrator.process(frame)
    if result["ready"]:
        H         = result["homography"]
        px_per_mm = result["px_per_mm"]
        mm_coords = calibrator.pixel_to_mm(px, py)
"""

import cv2
import numpy as np
from cv2 import aruco
from collections import deque
from typing import Optional, Dict, Tuple

from config import settings
import json
import os


# ── Constants ────────────────────────────────────────────────────────────────
MAT_IDS        = settings.MAT_IDS        # [4, 0, 2, 3] clockwise TL,TR,BR,BL
SMOOTH_FRAMES  = 15                       # rolling buffer for stability
MARKER_SIZE_MM = 16.0                     # physical ArUco marker size in mm

# Real-world corners in mm — order matches MAT_IDS clockwise
# Origin = ID4 (top-left), x→ right, y↓ down
MAT_CORNERS_MM = np.array([
    [0,                      0                     ],  # ID=0 top-left
    [settings.MAT_WIDTH_MM,  0                     ],  # ID=2 top-right
    [settings.MAT_WIDTH_MM,  settings.MAT_HEIGHT_MM],  # ID=3 bottom-right
    [0,                      settings.MAT_HEIGHT_MM],  # ID=4 bottom-left
], dtype=np.float32)

class MatCalibrator:
    """
    Detects the 4 mat corner ArUco markers and computes the homography
    that maps camera pixels to real-world mm coordinates.

    Call process(frame) every frame. Once ready=True the homography
    is stable and only recomputes if a marker is lost and reacquired.

    All status text is rendered by PyQt (in main_window._update_aruco_label).
    draw_overlay() only draws geometry — border and corner dots.
    """

    def __init__(self):
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self._detector   = aruco.ArucoDetector(
            self._aruco_dict, aruco.DetectorParameters()
        )

        # Rolling buffers for each corner — smooths jitter
        self._buffers: Dict[int, deque] = {
            mid: deque(maxlen=SMOOTH_FRAMES) for mid in MAT_IDS
        }

        # Latest computed result
        self._homography: Optional[np.ndarray] = None
        self._px_per_mm:  Optional[float]      = None
        self._ready:      bool                 = False
        self._px_per_mm = self._load_spatial_scale()  # load once at startup

    


    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> dict:
        """
        Run mat marker detection on a frame.

        Args:
            frame: BGR image from camera.

        Returns:
            State dict with at minimum {"ready": bool}.
            When ready=True also contains:
                homography   (3x3 np.ndarray)
                px_per_mm    (float)
                corners_px   (dict: id → [x, y] smoothed pixel position)
                fills        (dict: id → int, buffer fill count)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        # Update buffers
        fills = {mid: len(self._buffers[mid]) for mid in MAT_IDS}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid in self._buffers:
                    center = self._marker_center(corners[i])
                    self._buffers[mid].append(center)
                    fills[mid] = len(self._buffers[mid])

        # Check all 4 corners have data
        smoothed = {mid: self._smoothed(mid) for mid in MAT_IDS}
        if any(pt is None for pt in smoothed.values()):
            self._ready = False
            return {"ready": False, "fills": fills}

        # Source points in clockwise order: TL, TR, BR, BL
        src_pts = np.array(
            [smoothed[mid] for mid in MAT_IDS], dtype=np.float32
        )

        H, _ = cv2.findHomography(src_pts, MAT_CORNERS_MM)

        if H is None:
            self._ready = False
            return {"ready": False, "fills": fills}

        self._homography = H


        # Compute px_per_mm — use saved spatial calibration if available
        if ids is not None:
            mat_corners_raw = [
                corners[i]
                for i, mid in enumerate(ids.flatten())
                if mid in self._buffers
            ]
            if mat_corners_raw:
                if self._px_per_mm is None:
                    self._px_per_mm = self._estimate_px_per_mm(mat_corners_raw) 


        # Print to terminal once when first ready
        if not self._ready:
            print(f"✓ Mat calibrated — {self._px_per_mm:.3f} px/mm | "
                  f"{settings.MAT_WIDTH_MM:.0f}×{settings.MAT_HEIGHT_MM:.0f}mm")

        self._ready = True

        return {
            "ready":      True,
            "homography": H,
            "px_per_mm":  self._px_per_mm,
            "corners_px": smoothed,
            "fills":      fills,
        }

    def pixel_to_mm(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """
        Convert a pixel coordinate to real-world mm using the homography.
        """
        if self._homography is None:
            return None
        pt     = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._homography)
        return float(result[0][0][0]), float(result[0][0][1])

    def mm_to_pixel(self, x_mm: float, y_mm: float) -> Optional[Tuple[float, float]]:
        """
        Convert real-world mm back to pixel coordinates (inverse homography).
        """
        if self._homography is None:
            return None
        H_inv  = np.linalg.inv(self._homography)
        pt     = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H_inv)
        return float(result[0][0][0]), float(result[0][0][1])

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._homography

    @property
    def px_per_mm(self) -> Optional[float]:
        return self._px_per_mm

    # ── Overlay — geometry only, no text ─────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw mat calibration overlay — geometry only.
        No text is drawn here — all text is rendered by PyQt in the UI layer.

        Draws:
            - Mat border rectangle (green)
            - Corner dots (colored by position)
        """
        if not self._ready:
            return frame

        smoothed = {mid: self._smoothed(mid) for mid in MAT_IDS}

        corner_colors = {
            4: (255, 255,   0),   # TL yellow
            0: (255, 165,   0),   # TR orange
            2: (  0, 200, 255),   # BR cyan
            3: (200,   0, 255),   # BL purple
        }

        # Mat border — clockwise TL→TR→BR→BL
        pts = np.array(
            [smoothed[mid].astype(int) for mid in MAT_IDS], dtype=np.int32
        )
        cv2.polylines(frame, [pts], isClosed=True,
                      color=(0, 255, 100), thickness=2)

        # Corner dots only — no labels
        for mid in MAT_IDS:
            pt = smoothed[mid]
            if pt is None:
                continue
            ix, iy = int(pt[0]), int(pt[1])
            cv2.circle(frame, (ix, iy), 7, corner_colors[mid], -1)
            cv2.circle(frame, (ix, iy), 7, (255, 255, 255), 1)

        return frame

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _marker_center(corners) -> np.ndarray:
        c = corners[0]
        return np.array([c[:, 0].mean(), c[:, 1].mean()], dtype=np.float32)

    def _smoothed(self, mid: int) -> Optional[np.ndarray]:
        buf = self._buffers.get(mid)
        if buf and len(buf) > 0:
            return np.mean(buf, axis=0).astype(np.float32)
        return None

    @staticmethod
    def _estimate_px_per_mm(mat_corners_raw) -> float:
        """
        Compute px/mm from the physical size of the ArUco markers.
        Each marker is MARKER_SIZE_MM x MARKER_SIZE_MM.
        Averages across all detected mat markers for stability.
        """
        side_lengths = []
        for corner in mat_corners_raw:
            c = corner[0]  # shape (4, 2)
            sides = [
                np.linalg.norm(c[1] - c[0]),
                np.linalg.norm(c[2] - c[1]),
                np.linalg.norm(c[3] - c[2]),
                np.linalg.norm(c[0] - c[3]),
            ]
            side_lengths.append(float(np.mean(sides)))

        avg_side_px = float(np.mean(side_lengths))
        return avg_side_px / MARKER_SIZE_MM
    

    def _load_spatial_scale(self) -> Optional[float]:
        import json
        path = "calibration/spatial_scale.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            val = data.get("corrected_px_per_mm") or data.get("px_per_mm")
            if val:
                print(f"  ✓ [MatCalibrator] Spatial scale loaded: {float(val):.4f} px/mm")
                return float(val)
        return None
