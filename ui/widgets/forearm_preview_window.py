"""
Forearm camera preview window.

Displays the live forearm camera feed in a separate window with:
- ArUco geometry overlay (from mat_calibrator — corner dots + border)
- 10mm grid lines drawn by PyQt (crisp at any resolution)
- All text drawn by PyQt (corner labels, scale readout, status)

Separating text and grid from OpenCV means they scale correctly
regardless of camera resolution vs display size.

Usage:
    window = ForearmPreviewWindow(forearm_camera)
    window.show()
    window.close()
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont,
                          QFontMetrics)
from PyQt6.QtCore import QTimer, Qt, QPoint, QRect
from typing import Optional, Tuple

from config import settings


class ForearmPreviewWindow(QWidget):
    """
    Separate window showing the live forearm camera feed.

    Pulls frames from ForearmCamera at ~30 FPS using a QTimer.
    Draws grid and text using QPainter on top of the camera frame.
    """

    # Grid settings
    GRID_SPACING_MM  = 10.0
    GRID_MAJOR_EVERY = 20   # draw label every N mm
    GRID_MINOR_COLOR = QColor(0, 0, 0, 60)    # subtle black
    GRID_MAJOR_COLOR = QColor(0, 0, 0, 120)   # stronger black

    # Text settings
    LABEL_FONT_SIZE  = 11
    STATUS_FONT_SIZE = 13

    def __init__(self, camera, parent=None):
        """
        Args:
            camera: ForearmCamera instance (must be started).
            parent: Parent QWidget.
        """
        super().__init__(parent)
        self.camera = camera

        self.setWindowTitle("Forearm Camera — EMS")
        self.setMinimumSize(800, 500)
        self.resize(1000, 620)
        self.setStyleSheet("background-color: #1a1a1a;")

        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Canvas — we paint everything here
        self._canvas = _ForearmCanvas(self)
        layout.addWidget(self._canvas)

        # Timer ~30 FPS
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(33)

    def _update(self) -> None:
        """Pull latest frame + state and push to canvas."""
        frame = self.camera.get_frame()
        state = self.camera.get_aruco_state()
        self._canvas.update_data(frame, state)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        event.accept()

    def stop(self) -> None:
        self._timer.stop()


class _ForearmCanvas(QWidget):
    """
    Internal canvas widget that renders the camera frame and all overlays.

    Rendering order:
        1. Camera frame (scaled to fit)
        2. Grid lines (QPainter)
        3. Corner labels + scale text (QPainter)
        4. Status text if not ready (QPainter)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._state:  dict              = {"ready": False}

        # Scale factors — updated when frame size changes
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._frame_w:  int   = 1
        self._frame_h:  int   = 1

    def update_data(self, frame: Optional[np.ndarray], state: dict) -> None:
        """Receive new frame + state and trigger repaint."""
        if frame is not None:
            self._pixmap = self._cv_to_pixmap(frame)
            self._frame_w = frame.shape[1]
            self._frame_h = frame.shape[0]
        self._state = state
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ── 1. Draw camera frame ──────────────────────────────────────────────
        if self._pixmap:
            w, h = self.width(), self.height()
            scaled = self._pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            # Center in widget
            self._offset_x = (w - scaled.width())  / 2.0
            self._offset_y = (h - scaled.height()) / 2.0
            self._scale_x  = scaled.width()  / self._frame_w
            self._scale_y  = scaled.height() / self._frame_h

            painter.drawPixmap(int(self._offset_x), int(self._offset_y), scaled)
        else:
            painter.fillRect(self.rect(), QColor(30, 30, 30))

        if not self._state.get("ready"):
            self._draw_status(painter)
            return

        # ── 2. Grid lines ─────────────────────────────────────────────────────
        self._draw_grid(painter)

        # ── 3. Corner labels + scale readout ──────────────────────────────────
        self._draw_labels(painter)

    # ── Grid ──────────────────────────────────────────────────────────────────

    def _draw_grid(self, painter: QPainter) -> None:
        """Draw 10mm grid using homography to find pixel positions."""
        H = self._state.get("homography")
        if H is None:
            return

        spacing = ForearmPreviewWindow.GRID_SPACING_MM
        major_every = ForearmPreviewWindow.GRID_MAJOR_EVERY

        font = QFont("Helvetica", 9)
        painter.setFont(font)

        # ── Vertical lines ────────────────────────────────────────────────────
        x = spacing
        while x < settings.MAT_WIDTH_MM:
            pt_top = self._mm_to_display(H, x, 0)
            pt_bot = self._mm_to_display(H, x, settings.MAT_HEIGHT_MM)

            if pt_top and pt_bot:
                is_major = (round(x) % major_every == 0)
                pen = QPen(
                    ForearmPreviewWindow.GRID_MAJOR_COLOR if is_major
                    else ForearmPreviewWindow.GRID_MINOR_COLOR
                )
                pen.setWidthF(1.0 if is_major else 0.5)
                painter.setPen(pen)
                painter.drawLine(
                    QPoint(int(pt_top[0]), int(pt_top[1])),
                    QPoint(int(pt_bot[0]),  int(pt_bot[1])),
                )

                # Label at top
                if is_major:
                    label_pt = self._mm_to_display(H, x, -8)   # 8 mm "above" mat in mat frame
                    if label_pt:
                        self._draw_grid_label(
                            painter, f"{int(x)}",
                            int(label_pt[0]), int(label_pt[1]),
                            align="center"
                        )
            x += spacing

        # ── Horizontal lines ──────────────────────────────────────────────────
        y = spacing
        while y < settings.MAT_HEIGHT_MM:
            pt_left  = self._mm_to_display(H, 0,                     y)
            pt_right = self._mm_to_display(H, settings.MAT_WIDTH_MM, y)

            if pt_left and pt_right:
                is_major = (round(y) % major_every == 0)
                pen = QPen(
                    ForearmPreviewWindow.GRID_MAJOR_COLOR if is_major
                    else ForearmPreviewWindow.GRID_MINOR_COLOR
                )
                pen.setWidthF(1.0 if is_major else 0.5)
                painter.setPen(pen)
                painter.drawLine(
                    QPoint(int(pt_left[0]),  int(pt_left[1])),
                    QPoint(int(pt_right[0]), int(pt_right[1])),
                )

                # Label at left
                if is_major:
                    label_pt = self._mm_to_display(H, -8, y)   # 8 mm "left" of mat in mat frame
                    if label_pt:
                        self._draw_grid_label(
                            painter, f"{int(y)}",
                            int(label_pt[0]), int(label_pt[1]),
                            align="right"
                        )
            y += spacing

    def _draw_grid_label(self, painter: QPainter, text: str,
                          x: int, y: int, align: str = "center") -> None:
        """Draw a small grid label with white background."""
        font = QFont("Helvetica", 8)
        painter.setFont(font)
        fm   = QFontMetrics(font)
        tw   = fm.horizontalAdvance(text)
        th   = fm.height()

        if align == "center":
            rx = x - tw // 2 - 2
        else:  # right
            rx = x - tw - 4
        ry = y - th // 2

        # White background
        painter.fillRect(QRect(rx - 1, ry - 1, tw + 6, th + 2),
                         QColor(255, 255, 255, 180))
        painter.setPen(QPen(QColor(60, 60, 60)))
        painter.drawText(QPoint(rx + 2, ry + th - 2), text)

    # ── Corner labels ─────────────────────────────────────────────────────────

    def _draw_labels(self, painter: QPainter) -> None:
        """Draw corner labels and scale readout."""
        corners = self._state.get("corners_px", {})
        px_per_mm = self._state.get("px_per_mm")

        corner_info = {
            0: ("TL(0)", QColor(255, 165,   0)),  # orange
            2: ("TR(2)", QColor(  0, 200, 255)),  # cyan
            3: ("BR(3)", QColor(200,   0, 255)),  # purple
            4: ("BL(4)", QColor(255, 255,   0)),  # yellow
        }

        font = QFont("Helvetica", ForearmPreviewWindow.LABEL_FONT_SIZE,
                     QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        for mid, (label, color) in corner_info.items():
            pt = corners.get(mid)
            if pt is None:
                continue
            dx, dy = self._cam_to_display(float(pt[0]), float(pt[1]))
            if dx is None:
                continue

            # Label background
            tw = fm.horizontalAdvance(label)
            th = fm.height()
            rx, ry = int(dx) + 12, int(dy) - th - 4
            painter.fillRect(QRect(rx - 2, ry - 2, tw + 8, th + 4),
                             QColor(0, 0, 0, 180))

            # Label text
            painter.setPen(QPen(color))
            painter.drawText(QPoint(rx + 2, ry + th - 2), label)

        # Scale readout — bottom-left of image area
        if px_per_mm:
            lines = [
                f"Mat: {settings.MAT_WIDTH_MM:.0f}×{settings.MAT_HEIGHT_MM:.0f} mm",
                f"Scale: {px_per_mm:.3f} px/mm",
            ]
            font2 = QFont("Helvetica", ForearmPreviewWindow.STATUS_FONT_SIZE)
            painter.setFont(font2)
            fm2   = QFontMetrics(font2)
            x0    = int(self._offset_x) + 10
            y0    = int(self._offset_y + self._frame_h * self._scale_y) - 10

            for i, line in enumerate(reversed(lines)):
                tw = fm2.horizontalAdvance(line)
                th = fm2.height()
                ry = y0 - i * (th + 4)
                painter.fillRect(QRect(x0 - 2, ry - th - 2, tw + 8, th + 4),
                                 QColor(0, 0, 0, 180))
                painter.setPen(QPen(QColor(0, 255, 100)))
                painter.drawText(QPoint(x0 + 2, ry - 2), line)

    # ── Status (not ready) ────────────────────────────────────────────────────

    def _draw_status(self, painter: QPainter) -> None:
        """Draw status overlay when mat is not calibrated."""
        fills     = self._state.get("fills", {})
        mat_ready = self._state.get("mat_ready", False)

        font = QFont("Helvetica", ForearmPreviewWindow.STATUS_FONT_SIZE)
        painter.setFont(font)
        fm = QFontMetrics(font)

        if not mat_ready:
            # Show which mat corners are detected
            labels = {0: "TL", 2: "TR", 3: "BR", 4: "BL"}
            colors = {0: QColor(255,165,0), 2: QColor(0,200,255),
                    3: QColor(200,0,255), 4: QColor(255,255,0)}
            ids    = [0, 2, 3, 4]

            x0, y0 = int(self._offset_x) + 12, int(self._offset_y) + 28
            for i, mid in enumerate(ids):
                count   = fills.get(mid, 0)
                filled  = int((count / 15) * 12)
                bar     = f"Mat{mid}({labels[mid]}): [{'|'*filled}{'.'*(12-filled)}]  {count}/15"
                color   = QColor(0, 255, 0) if count == 15 else colors[mid]
                tw      = fm.horizontalAdvance(bar)
                th      = fm.height()
                ry      = y0 + i * (th + 6)
                painter.fillRect(QRect(x0 - 2, ry - th, tw + 8, th + 4),
                                 QColor(0, 0, 0, 180))
                painter.setPen(QPen(color))
                painter.drawText(QPoint(x0 + 2, ry), bar)

            # Main status message
            msg  = "Waiting for mat corners (ID 0, 2, 3, 4)..."
            font2 = QFont("Helvetica", 15, QFont.Weight.Bold)
            painter.setFont(font2)
            fm2  = QFontMetrics(font2)
            tw   = fm2.horizontalAdvance(msg)
            cx   = (self.width() - tw) // 2
            cy   = self.height() // 2
            painter.fillRect(QRect(cx - 8, cy - 24, tw + 16, 36),
                             QColor(0, 0, 0, 200))
            painter.setPen(QPen(QColor(0, 80, 255)))
            painter.drawText(QPoint(cx, cy), msg)

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _cam_to_display(self, cam_x: float, cam_y: float
                         ) -> Tuple[Optional[float], Optional[float]]:
        """Convert camera pixel → display pixel (accounts for scaling + offset)."""
        if self._scale_x == 0:
            return None, None
        dx = cam_x * self._scale_x + self._offset_x
        dy = cam_y * self._scale_y + self._offset_y
        return dx, dy

    def _mm_to_display(self, H: np.ndarray, x_mm: float, y_mm: float
                        ) -> Optional[Tuple[float, float]]:
        """Convert mm (mat frame) → display pixel via inverse homography + scale."""
        H_inv  = np.linalg.inv(H)
        pt     = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H_inv)
        cam_x  = float(result[0][0][0])
        cam_y  = float(result[0][0][1])
        dx, dy = self._cam_to_display(cam_x, cam_y)
        if dx is None:
            return None
        return dx, dy

    @staticmethod
    def _cv_to_pixmap(frame: np.ndarray) -> QPixmap:
        """Convert BGR OpenCV frame to QPixmap."""
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch     = rgb.shape
        bytes_per_line = ch * w
        q_image      = QImage(rgb.data, w, h, bytes_per_line,
                               QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)
