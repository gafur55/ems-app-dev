"""
Anatomical coordinate mapping system.
Converts pixel coordinates to anatomical coordinates (s, t).

Coordinate System
-----------------
    s (longitudinal): 0 = wrist, 1 = elbow
    t (lateral):      normalized by forearm width, radial/thumb side = positive

Two construction paths
----------------------
1. Manual (existing):
       AnatomicalCoordinateMapper(wrist, elbow, radial, u, v, ...)
   Used by the 3-click CalibrationManager workflow.

2. ArUco (new):
       AnatomicalCoordinateMapper.from_aruco_state(state)
   Built automatically from ArucoTracker state when all 4 markers
   are detected. No manual measurement or clicking required.
"""

import numpy as np
import math
from typing import Dict, Any


class AnatomicalCoordinateMapper:
    """
    Maps between pixel coordinates and anatomical normalized coordinates.

    Coordinate System:
        s (longitudinal): 0 = wrist, 1 = elbow
        t (lateral):      normalized, radial/thumb side = positive
    """

    def __init__(
        self,
        wrist,
        elbow,
        radial,
        u,
        v,
        forearm_length_px,
        forearm_length_cm,
        calibration_factor,
    ):
        """
        Initialize anatomical coordinate mapper.

        Args:
            wrist:               Wrist center position (numpy array [x, y])
            elbow:               Elbow center position (numpy array [x, y])
            radial:              Radial styloid / thumb-side reference (numpy array [x, y])
            u:                   Unit vector along forearm (wrist → elbow)
            v:                   Unit vector perpendicular (radial/thumb side = positive)
            forearm_length_px:   Forearm length in pixels
            forearm_length_cm:   Forearm length in centimetres
            calibration_factor:  Pixels per centimetre
        """
        self.wrist              = np.asarray(wrist,  dtype=float)
        self.elbow              = np.asarray(elbow,  dtype=float)
        self.radial             = np.asarray(radial, dtype=float)
        self.u                  = np.asarray(u,      dtype=float)
        self.v                  = np.asarray(v,      dtype=float)
        self.forearm_length_px  = float(forearm_length_px)
        self.forearm_length_cm  = float(forearm_length_cm)
        self.calibration_factor = float(calibration_factor)  # px/cm

        # Forearm width estimate — used for t normalisation
        radial_offset          = self.radial - self.wrist
        self.estimated_width_px = abs(np.dot(radial_offset, self.v)) * 2
        self.estimated_width_cm = self.estimated_width_px / self.calibration_factor

    # ── ArUco factory ─────────────────────────────────────────────────────────

    @classmethod
    def from_aruco_state(cls, state: Dict[str, Any]) -> "AnatomicalCoordinateMapper":
        """
        Build a mapper directly from an ArucoTracker state dict.

        Uses the 4-corner markers to derive all axes and scale factors
        automatically — no manual measurements or forearm length input needed.

        Args:
            state: Dict returned by ArucoTracker.get_state() with ready=True.

        Returns:
            AnatomicalCoordinateMapper ready for pixel_to_anatomical() calls.

        Raises:
            ValueError: if state["ready"] is False.
        """
        if not state.get("ready"):
            raise ValueError("ArUco state is not ready — all 4 reference markers must be visible.")

        wrist_center = np.asarray(state["wrist_center_px"], dtype=float)
        elbow_center = np.asarray(state["elbow_center_px"], dtype=float)
        wrist_thumb  = np.asarray(state["wrist_thumb_px"],  dtype=float)
        wrist_pinky  = np.asarray(state["wrist_pinky_px"],  dtype=float)

        px_per_mm = float(state["px_per_mm"])
        px_per_cm = px_per_mm * 10.0

        # Longitudinal axis: wrist → elbow (s increases toward elbow)
        forearm_vec       = elbow_center - wrist_center
        forearm_length_px = float(np.linalg.norm(forearm_vec))
        u                 = forearm_vec / forearm_length_px

        # Lateral axis: pinky → thumb at wrist, perpendicular to u
        lat_raw = wrist_thumb - wrist_pinky
        lat_raw -= np.dot(lat_raw, u) * u        # make perpendicular
        v        = lat_raw / np.linalg.norm(lat_raw)   # thumb side = positive

        # Use wrist_thumb as the radial/thumb reference point
        radial = wrist_thumb

        forearm_length_cm = float(state["forearm_length_mm"]) / 10.0

        return cls(
            wrist              = wrist_center,
            elbow              = elbow_center,
            radial             = radial,
            u                  = u,
            v                  = v,
            forearm_length_px  = forearm_length_px,
            forearm_length_cm  = forearm_length_cm,
            calibration_factor = px_per_cm,
        )

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def pixel_to_anatomical(self, pixel_x: float, pixel_y: float):
        """
        Convert pixel coordinates to anatomical normalised coordinates.

        Args:
            pixel_x: X position in image
            pixel_y: Y position in image

        Returns:
            Tuple (s, t, real_x_cm, real_y_cm) where:
                s:          longitudinal (0 = wrist, 1 = elbow)
                t:          lateral (normalised, thumb/radial = positive)
                real_x_cm:  lateral distance in cm  (thumb = positive)
                real_y_cm:  longitudinal distance in cm (toward elbow = positive)
        """
        point            = np.array([pixel_x, pixel_y], dtype=float)
        electrode_vector = point - self.wrist

        longitudinal_px = np.dot(electrode_vector, self.u)
        s               = longitudinal_px / self.forearm_length_px

        lateral_px = np.dot(electrode_vector, self.v)
        t          = lateral_px / self.estimated_width_px if self.estimated_width_px > 0 else 0.0

        real_y_cm = longitudinal_px / self.calibration_factor
        real_x_cm = lateral_px      / self.calibration_factor

        return s, t, real_x_cm, real_y_cm

    def anatomical_to_pixel(self, s: float, t: float):
        """
        Convert anatomical coordinates back to pixel coordinates.

        Args:
            s: longitudinal (0 = wrist, 1 = elbow)
            t: lateral (normalised)

        Returns:
            Tuple (pixel_x, pixel_y)
        """
        longitudinal_px = s * self.forearm_length_px
        lateral_px      = t * self.estimated_width_px
        point           = self.wrist + longitudinal_px * self.u + lateral_px * self.v
        return float(point[0]), float(point[1])

    def format_coordinates(self, s, t, real_x_cm, real_y_cm) -> str:
        """Human-readable coordinate string."""
        percentage = s * 100

        if real_y_cm < 0:
            longitudinal = (f"{abs(real_y_cm):.1f}cm proximal to wrist "
                            f"({abs(percentage):.0f}% before wrist)")
        else:
            longitudinal = (f"{real_y_cm:.1f}cm from wrist "
                            f"({percentage:.0f}% along forearm)")

        if t > 0:
            lateral = f"{abs(real_x_cm):.1f}cm radial side (thumb side)"
        elif t < 0:
            lateral = f"{abs(real_x_cm):.1f}cm ulnar side (pinky side)"
        else:
            lateral = "on centreline"

        distance = self.distance_from_origin(real_x_cm, real_y_cm)
        return (f"{longitudinal}, {lateral}\n"
                f"  Normalised: (s={s:.3f}, t={t:.3f})\n"
                f"  Distance from wrist: {distance:.1f}cm")

    @staticmethod
    def distance_from_origin(real_x_cm: float, real_y_cm: float) -> float:
        """Euclidean distance from wrist origin."""
        return math.sqrt(real_x_cm ** 2 + real_y_cm ** 2)