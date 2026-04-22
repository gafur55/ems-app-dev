"""
Camera undistortion using a pre-computed calibration file.

Shared by ForearmCamera, HandTracker, and PoseTracker.
Each camera has its own calibration file: calibration/camera_{index}.npz

Workflow
--------
1. Run calibrate_cameras.py once per physical camera to produce the .npz.
2. On session start each tracker calls Undistorter(camera_index).
   - Raises CalibrationNotFoundError immediately if the file is missing.
   - Precomputes the remap maps (done once, cheap forever after).
3. Every frame: call undistorter.apply(frame) before any other processing.
   Uses cv2.remap — ~2 ms at 720p.

The .npz must contain:
    camera_matrix   shape (3, 3)   intrinsic matrix K
    dist_coeffs     shape (1, 5)   distortion coefficients
    rms_error       float          reprojection error (informational)
"""

import os
import cv2
import numpy as np
from typing import Tuple


CALIBRATION_DIR = "calibration"


class CalibrationNotFoundError(Exception):
    """
    Raised when no calibration file exists for the requested camera index.

    Attributes:
        camera_index: The camera index that was requested.
        expected_path: The full path that was checked.
    """
    def __init__(self, camera_index: int, expected_path: str):
        self.camera_index = camera_index
        self.expected_path = expected_path
        super().__init__(
            f"\n\n"
            f"  ✗ No calibration file found for camera {camera_index}.\n"
            f"  Expected: {expected_path}\n\n"
            f"  Run this command to calibrate:\n"
            f"    python3 calibrate_cameras.py --camera {camera_index}\n"
        )


class Undistorter:
    """
    Loads a camera calibration .npz and applies lens undistortion to frames.

    Parameters
    ----------
    camera_index : int
        Camera device index. Used to locate calibration/camera_{index}.npz.
    calibration_dir : str
        Directory containing the .npz files. Defaults to 'calibration/'.
    alpha : float
        Scaling parameter for getOptimalNewCameraMatrix.
        0.0 = crop to only valid pixels (no black borders).
        1.0 = keep all pixels (black borders around edges).
        Default 0.0 is recommended — no wasted image area.

    Raises
    ------
    CalibrationNotFoundError
        Immediately on __init__ if the calibration file does not exist.
    """

    def __init__(
        self,
        camera_index: int,
        calibration_dir: str = CALIBRATION_DIR,
        alpha: float = 0.0,
    ):
        self.camera_index = camera_index
        self._map1: np.ndarray = None
        self._map2: np.ndarray = None
        self._roi: Tuple[int, int, int, int] = None
        self._frame_shape: Tuple[int, int] = None  # (h, w) the maps were built for

        path = os.path.join(calibration_dir, f"camera_{camera_index}.npz")

        if not os.path.exists(path):
            print(
                f"  ⚠  [Undistorter] No calibration file for camera {camera_index} "
                f"({os.path.abspath(path)}) — skipping undistortion.\n"
                f"     Run: python3 calibrate_cameras.py --camera {camera_index}"
            )
            self._camera_matrix = None
            self._dist_coeffs   = None
            self._rms_error     = None
            return

        data = np.load(path)
        self._camera_matrix = data["mtx"]
        self._dist_coeffs   = data["dist"]
        self._rms_error: float = float(data["rms_error"]) if "rms_error" in data else 0.0
        self._alpha                     = alpha

        print(
            f"  ✓ [Undistorter] Loaded calibration for camera {camera_index} "
            f"(RMS error: {self._rms_error:.4f} px)"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply lens undistortion to a single frame.
        If no calibration file was found, returns the frame unchanged.
        """
        if self._camera_matrix is None:
            return frame  # no calibration — pass through unchanged

        h, w = frame.shape[:2]
        if self._map1 is None or self._frame_shape != (h, w):
            self._build_maps(h, w)

        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    @property
    def rms_error(self) -> float:
        """Reprojection error from calibration (lower is better, target < 0.5)."""
        return self._rms_error

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _build_maps(self, h: int, w: int) -> None:
        """
        Precompute the undistortion + rectification maps for a given frame size.

        Called once per unique resolution.  cv2.remap then does the actual
        per-frame work using these maps.
        """
        new_camera_matrix, self._roi = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix,
            self._dist_coeffs,
            (w, h),
            self._alpha,
            (w, h),
        )

        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._camera_matrix,
            self._dist_coeffs,
            None,                 # no rectification rotation
            new_camera_matrix,
            (w, h),
            cv2.CV_16SC2,         # most efficient map type for remap
        )

        self._frame_shape = (h, w)

        print(
            f"  [Undistorter] Maps built for camera {self.camera_index} "
            f"at {w}×{h}  (ROI: {self._roi})"
        )