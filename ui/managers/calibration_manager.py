"""
ArUco-based calibration manager.

The manual 3-click workflow (wrist → elbow → radial) has been removed.
Calibration now happens automatically whenever all 4 ArUco reference
markers are visible in the forearm camera feed.

All it does:
    1. Receive an ArUco state dict from ForearmCamera.get_aruco_state()
    2. Build an AnatomicalCoordinateMapper from it
    3. Draw the detected marker positions on the scene
    4. Emit calibration_complete(mapper)

The forearm camera's live annotated feed (with grid, axes, anatomy HUD)
IS the calibration UI — no clicks required.
"""

from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsLineItem,
)
from PyQt6.QtGui  import QBrush, QPen, QColor, QFont
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from typing import List
import numpy as np


class CalibrationManager(QObject):
    """
    ArUco-only calibration manager.

    Signals:
        calibration_complete: emitted with AnatomicalCoordinateMapper when done.
    """

    calibration_complete = pyqtSignal(object)

    def __init__(self, scene: QGraphicsScene):
        super().__init__()
        self.scene        = scene
        self.session      = None
        self._scene_items: List = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """Always False — no manual click workflow exists anymore."""
        return False

    # ── Session ───────────────────────────────────────────────────────────────

    def set_session(self, session) -> None:
        self.session = session

    # ── ArUco calibration ─────────────────────────────────────────────────────

    def complete_from_aruco(self, aruco_state: dict) -> bool:
        """
        Build coordinate mapper from ArUco state and emit signal.

        Args:
            aruco_state: dict from ForearmCamera.get_aruco_state() with ready=True.

        Returns:
            True on success, False if state not ready.
        """
        if not aruco_state.get("ready"):
            print("✗ [Calibration] ArUco not ready — ensure all 4 markers are visible.")
            return False

        from core.vision.anatomical_mapper import AnatomicalCoordinateMapper
        mapper = AnatomicalCoordinateMapper.from_aruco_state(aruco_state)

        if self.session:
            wrist_px = aruco_state["wrist_center_px"]
            self.session.origin_pixel_x    = float(wrist_px[0])
            self.session.origin_pixel_y    = float(wrist_px[1])
            self.session.calibration_factor = mapper.calibration_factor

        self._print_summary(aruco_state, mapper)
        self._draw_markers(aruco_state)
        self.calibration_complete.emit(mapper)
        return True

    def _print_summary(self, state: dict, mapper) -> None:
        print("\n=== ArUco Calibration ===")
        print(f"  Forearm : {state['forearm_length_mm']:.1f} mm")
        print(f"  Wrist W : {state['wrist_width_mm']:.1f} mm")
        print(f"  Elbow W : {state['elbow_width_mm']:.1f} mm")
        print(f"  Scale   : {state['px_per_mm']:.3f} px/mm")
        print("✓ Calibration complete — ready to place electrodes")

    def _draw_markers(self, state: dict) -> None:
        """Draw coloured reference dots on the Qt scene."""
        self.clear_markers()
        entries = [
            (state["wrist_center_px"], "WRIST",   QColor(255, 255,   0)),
            (state["elbow_center_px"], "ELBOW",   QColor(255,  80,  80)),
            (state["wrist_pinky_px"],  "W.Pinky", QColor(180, 180,   0)),
            (state["wrist_thumb_px"],  "W.Thumb", QColor(180, 180,   0)),
            (state["elbow_pinky_px"],  "E.Pinky", QColor(180,  80,  80)),
            (state["elbow_thumb_px"],  "E.Thumb", QColor(180,  80,  80)),
        ]
        for pos, label, color in entries:
            x, y   = float(pos[0]), float(pos[1])
            r      = 6
            circle = QGraphicsEllipseItem(x - r, y - r, r * 2, r * 2)
            circle.setBrush(QBrush(color))
            circle.setPen(QPen(Qt.GlobalColor.white, 1))
            self.scene.addItem(circle)
            self._scene_items.append(circle)

            text = QGraphicsTextItem(label)
            text.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            text.setDefaultTextColor(color)
            text.setPos(x + 8, y - 14)
            self.scene.addItem(text)
            self._scene_items.append(text)

        # Wrist–elbow axis line
        w = state["wrist_center_px"]
        e = state["elbow_center_px"]
        line = QGraphicsLineItem(float(w[0]), float(w[1]), float(e[0]), float(e[1]))
        line.setPen(QPen(QColor(0, 255, 100), 2))
        self.scene.addItem(line)
        self._scene_items.append(line)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def clear_markers(self) -> None:
        for item in self._scene_items:
            if item.scene() is not None:
                self.scene.removeItem(item)
        self._scene_items.clear()

    def reset(self) -> None:
        self.clear_markers()

    # ── Stub — kept so nothing breaks if called ───────────────────────────────

    def start(self, forearm_length_cm: float = 0) -> None:
        """No-op. Manual calibration removed."""
        print("  [CalibrationManager] Manual mode removed — ArUco only.")

    def handle_click(self, x: float, y: float) -> bool:
        """No-op."""
        return False