"""
ArUco-based forearm marker tracker.

Detects 4 reference markers on the forearm corners and up to N electrode
markers. Maintains rolling buffers for smooth positions and derives all
anatomical measurements automatically from the marker geometry.

Marker ID assignment
--------------------
    0  Wrist  Pinky edge
    1  Wrist  Thumb edge
    2  Elbow  Pinky edge  ← coordinate origin (0, 0)
    3  Elbow  Thumb edge
    4  Electrode A
    5  Electrode B
    (extend ELECTRODE_IDS as needed)

Coordinate system (all in mm)
------------------------------
    Origin  : elbow pinky corner (ID 2)
    Down    : elbow → wrist  (positive = toward wrist)
    Lateral : pinky → thumb  (positive = toward thumb)

Usage
-----
    tracker = ArucoTracker()
    # inside camera loop:
    annotated_frame, state = tracker.process(frame)
    if state["ready"]:
        print(state["wrist_width_mm"], state["forearm_length_mm"])
        for eid, info in state["electrodes"].items():
            print(eid, info["down_mm"], info["lateral_mm"])
"""

import cv2
import numpy as np
from cv2 import aruco
from collections import deque
from typing import Dict, Optional, Tuple

# ── Marker IDs ────────────────────────────────────────────────────────────────
WRIST_PINKY   = 0
WRIST_THUMB   = 1
ELBOW_PINKY   = 2   # origin (0, 0)
ELBOW_THUMB   = 3
ELECTRODE_IDS = [4, 5]

REFERENCE_IDS = [WRIST_PINKY, WRIST_THUMB, ELBOW_PINKY, ELBOW_THUMB]
ALL_IDS       = REFERENCE_IDS + ELECTRODE_IDS

# ── Physical constants ────────────────────────────────────────────────────────
MARKER_SIZE_MM  = 16.0   # printed ArUco square side length in mm
SMOOTH_FRAMES   = 10     # rolling window for position smoothing


class ArucoTracker:
    """
    Frame-by-frame ArUco marker detector with rolling-average smoothing.

    Call process(frame) on every camera frame. Returns an annotated frame
    and a state dict. Thread-safe if called from a single thread (the camera
    thread); state is read from the UI thread via get_state().
    """

    def __init__(
        self,
        marker_size_mm: float = MARKER_SIZE_MM,
        smooth_frames:  int   = SMOOTH_FRAMES,
        electrode_ids:  list  = None,
    ):
        self.marker_size_mm = marker_size_mm
        self.smooth_frames  = smooth_frames
        self.electrode_ids  = electrode_ids or ELECTRODE_IDS

        all_ids = REFERENCE_IDS + self.electrode_ids
        self._buffers:      Dict[int, deque] = {mid: deque(maxlen=smooth_frames) for mid in all_ids}
        self._ratio_buffer: deque            = deque(maxlen=smooth_frames)

        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self._detector   = aruco.ArucoDetector(self._aruco_dict, aruco.DetectorParameters())

        # Latest computed state (written by process(), read by get_state())
        self._state: dict = {"ready": False}

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Run ArUco detection on a frame.

        Args:
            frame: BGR image from camera (already undistorted)

        Returns:
            (annotated_frame, state_dict)
            annotated_frame has ArUco overlays drawn on it.
            state_dict is also stored as self._state.
        """
        annotated = frame.copy()
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self._detector.detectMarkers(gray)

        # ── Update buffers ────────────────────────────────────────────────────
        frame_side_pxs = []
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid in self._buffers:
                    self._buffers[mid].append(self._marker_center(corners[i]))
                frame_side_pxs.append(self._marker_side_px(corners[i]))
            aruco.drawDetectedMarkers(annotated, corners, ids)

        if frame_side_pxs:
            self._ratio_buffer.append(np.mean(frame_side_pxs) / self.marker_size_mm)

        # ── Compute state ─────────────────────────────────────────────────────
        state = self._compute_state(annotated)
        self._state = state
        return annotated, state

    def get_state(self) -> dict:
        """Return the most recently computed state (thread-safe read)."""
        return self._state

    def buffer_fill(self, marker_id: int) -> int:
        """Number of frames collected for a marker (0–smooth_frames)."""
        return len(self._buffers.get(marker_id, []))

    def is_reference_ready(self) -> bool:
        """True when all 4 reference markers have at least 1 frame."""
        return all(len(self._buffers[mid]) > 0 for mid in REFERENCE_IDS)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _marker_center(corners) -> np.ndarray:
        c = corners[0]
        return np.array([c[:, 0].mean(), c[:, 1].mean()])

    @staticmethod
    def _marker_side_px(corners) -> float:
        c = corners[0]
        sides = [
            np.linalg.norm(c[1] - c[0]),
            np.linalg.norm(c[2] - c[1]),
            np.linalg.norm(c[3] - c[2]),
            np.linalg.norm(c[0] - c[3]),
        ]
        return float(np.mean(sides))

    def _smoothed(self, mid: int) -> Optional[np.ndarray]:
        buf = self._buffers.get(mid)
        return np.mean(buf, axis=0) if buf and len(buf) > 0 else None

    def _px_per_mm(self) -> Optional[float]:
        return float(np.mean(self._ratio_buffer)) if self._ratio_buffer else None

    def _compute_state(self, annotated: np.ndarray) -> dict:
        """Build the full state dict from current buffer contents."""
        px_per_mm = self._px_per_mm()
        ref_pts   = [self._smoothed(mid) for mid in REFERENCE_IDS]

        # Buffer fill counts for HUD
        fills = {mid: self.buffer_fill(mid) for mid in REFERENCE_IDS}

        if any(pt is None for pt in ref_pts) or px_per_mm is None:
            self._draw_status(annotated, fills, px_per_mm)
            return {
                "ready":      False,
                "px_per_mm":  px_per_mm,
                "fills":      fills,
            }

        wp, wt, ep, et = ref_pts   # wrist-pinky, wrist-thumb, elbow-pinky, elbow-thumb

        # ── Derived anatomy ───────────────────────────────────────────────────
        wrist_center      = (wp + wt) / 2
        elbow_center      = (ep + et) / 2
        wrist_width_mm    = np.linalg.norm(wt - wp)               / px_per_mm
        elbow_width_mm    = np.linalg.norm(et - ep)               / px_per_mm
        forearm_length_mm = np.linalg.norm(wrist_center
                                           - elbow_center)          / px_per_mm

        # ── Coordinate axes ───────────────────────────────────────────────────
        origin, down_axis, lateral_axis = self._build_axes(
            wp, wt, ep, et, wrist_center, elbow_center)

        # ── Electrode positions ────────────────────────────────────────────────
        electrodes = {}
        for eid in self.electrode_ids:
            pt = self._smoothed(eid)
            if pt is None:
                continue
            down_mm, lat_mm = self._to_forearm_coords(
                pt, origin, down_axis, lateral_axis, px_per_mm)
            electrodes[eid] = {
                "pixel":       pt,
                "down_mm":     float(down_mm),
                "lateral_mm":  float(lat_mm),
                "side":        "thumb" if lat_mm >= 0 else "pinky",
                "buffer_full": self.buffer_fill(eid) == self.smooth_frames,
            }

        # ── Annotate frame ────────────────────────────────────────────────────
        self._draw_overlay(
            annotated, wp, wt, ep, et,
            wrist_center, elbow_center,
            origin, down_axis, lateral_axis,
            px_per_mm, forearm_length_mm, elbow_width_mm,
            wrist_width_mm, electrodes, fills,
        )

        state = {
            "ready":              True,
            "px_per_mm":          px_per_mm,
            "fills":              fills,
            # Pixel positions (smoothed)
            "wrist_pinky_px":     wp,
            "wrist_thumb_px":     wt,
            "elbow_pinky_px":     ep,
            "elbow_thumb_px":     et,
            "wrist_center_px":    wrist_center,
            "elbow_center_px":    elbow_center,
            "origin_px":          origin,           # = ep
            "down_axis":          down_axis,
            "lateral_axis":       lateral_axis,
            # Anatomy (mm)
            "wrist_width_mm":     float(wrist_width_mm),
            "elbow_width_mm":     float(elbow_width_mm),
            "forearm_length_mm":  float(forearm_length_mm),
            # Electrodes
            "electrodes":         electrodes,
        }
        return state

    @staticmethod
    def _build_axes(wp, wt, ep, et, wrist_center, elbow_center):
        """
        Returns (origin, down_axis, lateral_axis).
        Origin = elbow pinky corner.
        Down   = elbow → wrist (positive toward wrist).
        Lateral= pinky → thumb (positive toward thumb).
        """
        origin = ep.copy()

        down_vec  = wrist_center - elbow_center
        down_axis = down_vec / np.linalg.norm(down_vec)

        lat_wrist = wt - wp
        lat_elbow = et - ep
        lat_avg   = (lat_wrist + lat_elbow) / 2
        lat_avg  -= np.dot(lat_avg, down_axis) * down_axis
        lateral_axis = lat_avg / np.linalg.norm(lat_avg)

        return origin, down_axis, lateral_axis

    @staticmethod
    def _to_forearm_coords(point, origin, down_axis, lateral_axis, px_per_mm):
        delta      = point - origin
        down_mm    = np.dot(delta, down_axis)    / px_per_mm
        lateral_mm = np.dot(delta, lateral_axis) / px_per_mm
        return down_mm, lateral_mm

    # ── Drawing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _put_label(frame, text, pos, color=(0, 255, 0)):
        x, y = int(pos[0]) + 10, int(pos[1])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x - 2, y - th - 3), (x + tw + 2, y + 3), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_status(self, frame, fills, px_per_mm):
        """Draw buffer bars when not yet ready."""
        names = {
            WRIST_PINKY: "Wrist Pinky",
            WRIST_THUMB: "Wrist Thumb",
            ELBOW_PINKY: "Elbow Pinky",
            ELBOW_THUMB: "Elbow Thumb",
        }
        for row, (mid, name) in enumerate(names.items()):
            count  = fills.get(mid, 0)
            filled = int((count / self.smooth_frames) * 15)
            bar    = f"{name}: [{'|'*filled}{'.'*(15-filled)}] {count}/{self.smooth_frames}"
            color  = (0, 255, 0) if count == self.smooth_frames else (0, 165, 255)
            cv2.putText(frame, bar, (10, 25 + row * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if px_per_mm is None:
            cv2.putText(frame, "Waiting for markers (ID 0-3)...",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    def _draw_overlay(self, frame, wp, wt, ep, et,
                      wrist_c, elbow_c, origin, down_axis, lateral_axis,
                      px_per_mm, forearm_l, elbow_w, wrist_w,
                      electrodes, fills):
        """Draw full forearm overlay on the frame."""

        # Buffer bars
        self._draw_status(frame, fills, px_per_mm)

        # Forearm outline
        pts = np.array([wp, wt, et, ep], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(180, 180, 180), thickness=1)

        # Anatomy grid
        self._draw_grid(frame, origin, down_axis, lateral_axis,
                        px_per_mm, forearm_l, elbow_w)

        # Reference dots + labels
        for pt, name, color in [
            (wp,      "W.Pinky", (200, 200,   0)),
            (wt,      "W.Thumb", (200, 200,   0)),
            (ep,      "E.Pinky", (200, 100, 100)),
            (et,      "E.Thumb", (200, 100, 100)),
            (wrist_c, "WRIST",   (255, 255,   0)),
            (elbow_c, "ELBOW",   (255,  80,  80)),
        ]:
            cv2.circle(frame, tuple(pt.astype(int)), 5, color, -1)
            self._put_label(frame, name, pt, color)

        # Origin
        ox, oy = tuple(origin.astype(int))
        cv2.circle(frame, (ox, oy), 9, (0, 0, 255), -1)
        self._put_label(frame, "(0,0)", origin, (0, 0, 255))

        # Axes (50 mm long)
        def arrow_end(axis):
            return tuple((origin + axis * 50 * px_per_mm).astype(int))

        cv2.arrowedLine(frame, (ox, oy), arrow_end(down_axis),
                        (255, 60, 60), 2, tipLength=0.2)
        cv2.arrowedLine(frame, (ox, oy), arrow_end(lateral_axis),
                        (0, 165, 255), 2, tipLength=0.2)
        self._put_label(frame, "wrist",
                        origin + down_axis    * 52 * px_per_mm, (255, 60,  60))
        self._put_label(frame, "thumb",
                        origin + lateral_axis * 52 * px_per_mm, (  0, 165, 255))

        # Anatomy HUD (top-right)
        self._draw_anatomy_hud(frame, wrist_w, elbow_w, forearm_l, px_per_mm)

        # Electrode dots + labels
        origin_px = (ox, oy)
        for eid, info in electrodes.items():
            pt    = info["pixel"]
            px    = tuple(pt.astype(int))
            color = (0, 255, 0) if info["buffer_full"] else (0, 200, 200)
            thickness = -1 if info["buffer_full"] else 2
            cv2.line(frame, origin_px, px, (50, 50, 50), 1)
            cv2.circle(frame, px, 8, color, thickness)
            label = (f"E{eid}: {info['down_mm']:.1f}mm | "
                     f"{abs(info['lateral_mm']):.1f}mm {info['side']}")
            self._put_label(frame, label, pt)

    @staticmethod
    def _draw_grid(frame, origin, down_axis, lateral_axis, px_per_mm,
                   forearm_l_mm, elbow_w_mm, step_mm=20):
        down_steps = int(forearm_l_mm // step_mm) + 1
        lat_steps  = int(elbow_w_mm   // step_mm) + 1
        for d in range(down_steps + 1):
            d_mm = d * step_mm
            p1   = origin + down_axis    * d_mm * px_per_mm
            p2   = p1     + lateral_axis * elbow_w_mm * px_per_mm
            cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (55, 55, 55), 1)
            cv2.putText(frame, f"{d_mm}",
                        tuple((p1 - lateral_axis * 16).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (90, 90, 90), 1)
        for l in range(lat_steps + 1):
            l_mm = l * step_mm
            p1   = origin + lateral_axis * l_mm * px_per_mm
            p2   = p1     + down_axis    * forearm_l_mm * px_per_mm
            cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (55, 55, 55), 1)

    @staticmethod
    def _draw_anatomy_hud(frame, wrist_w, elbow_w, forearm_l, px_per_mm):
        lines = [
            f"Forearm : {forearm_l:.1f} mm",
            f"Wrist W : {wrist_w:.1f} mm",
            f"Elbow W : {elbow_w:.1f} mm",
            f"Scale   : {px_per_mm:.3f} px/mm",
        ]
        line_h  = 20
        padding = 6
        panel_w = 230
        panel_h = len(lines) * line_h + padding * 2
        x0      = frame.shape[1] - panel_w - 8
        y0      = 8
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line,
                        (x0 + padding, y0 + padding + (i + 1) * line_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)