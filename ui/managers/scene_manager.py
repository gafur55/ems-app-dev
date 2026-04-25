"""
Manages the graphics scene: setup, image loading, placeholders, live frame updates.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (QGraphicsScene, QGraphicsRectItem,
                              QGraphicsTextItem, QGraphicsPixmapItem,
                              QGraphicsItem)
from PyQt6.QtGui import QBrush, QPen, QColor, QPixmap, QImage, QPainter, QFont
from PyQt6.QtCore import Qt, QRectF, QPointF
from config import settings
from ui.managers.grid_overlay import GridOverlay
import os


# ── Homography grid overlay ───────────────────────────────────────────────────

class HomographyGridOverlay(QGraphicsItem):
    """
    Perspective-correct grid drawn using the mat homography.

    Grid lines are spaced GRID_SPACING_MM apart in real-world mm.
    The homography maps mm → camera pixels, which are then scaled
    to scene coordinates.

    Updated every frame via update_transform().
    """

    GRID_SPACING_MM  = 10.0
    MAJOR_EVERY_MM   = 20.0
    MINOR_COLOR      = QColor(0, 0, 0, 100)
    MAJOR_COLOR      = QColor(0, 0, 0, 200)

    def __init__(self):
        super().__init__()
        self._H_inv:    np.ndarray = None
        self._scale_x:  float     = 1.0
        self._scale_y:  float     = 1.0
        self._offset_x: float     = 0.0
        self._offset_y: float     = 0.0

        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable,    False)
        self.setZValue(-990)  # just above background, below electrodes

    def update_transform(self, H: np.ndarray,
                         scale_x: float, scale_y: float,
                         offset_x: float, offset_y: float) -> None:
        """Update homography and scale factors, then repaint."""
        try:
            self._H_inv    = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        self._scale_x  = scale_x
        self._scale_y  = scale_y
        self._offset_x = offset_x
        self._offset_y = offset_y
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0,
                      settings.BODY_DIAGRAM_WIDTH,
                      settings.BODY_DIAGRAM_HEIGHT)




    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._H_inv is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        spacing = self.GRID_SPACING_MM
        major   = self.MAJOR_EVERY_MM

        font = QFont("Helvetica", 9, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()

        # ── Vertical grid lines (constant x_mm, going from y=0 to y=MAT_HEIGHT) ──
        x = spacing
        while x < settings.MAT_WIDTH_MM:
            p1 = self._mm_to_scene(x, 0)
            p2 = self._mm_to_scene(x, settings.MAT_HEIGHT_MM)
            if p1 and p2:
                is_major = (round(x) % round(major) == 0)
                pen = QPen(self.MAJOR_COLOR if is_major else self.MINOR_COLOR)
                pen.setWidthF(1.5 if is_major else 0.8)
                painter.setPen(pen)
                painter.drawLine(p1, p2)

                # Label sits ~10mm OUTSIDE the bottom edge in MAT space.
                # The inverse homography places it correctly in scene space
                # regardless of how the camera is rotated.
                if is_major:
                    label_pt = self._mm_to_scene(x, settings.MAT_HEIGHT_MM + 10)
                    if label_pt:
                        text = f"{int(x)}"
                        tw   = fm.horizontalAdvance(text)
                        th   = fm.height()
                        painter.fillRect(
                            QRectF(label_pt.x() - tw/2 - 2, label_pt.y() - th + 2,
                                tw + 4, th),
                            QColor(255, 255, 255, 200)
                        )
                        painter.setPen(QPen(QColor(0, 0, 0)))
                        painter.drawText(
                            QPointF(label_pt.x() - tw/2, label_pt.y()),
                            text
                        )
            x += spacing

        # ── Horizontal grid lines (constant y_mm, going from x=0 to x=MAT_WIDTH) ─
        y = spacing
        while y < settings.MAT_HEIGHT_MM:
            p1 = self._mm_to_scene(0,                     y)
            p2 = self._mm_to_scene(settings.MAT_WIDTH_MM, y)
            if p1 and p2:
                is_major = (round(y) % round(major) == 0)
                pen = QPen(self.MAJOR_COLOR if is_major else self.MINOR_COLOR)
                pen.setWidthF(1.5 if is_major else 0.8)
                painter.setPen(pen)
                painter.drawLine(p1, p2)

                # Label sits ~10mm OUTSIDE the left edge in MAT space.
                if is_major:
                    label_pt = self._mm_to_scene(-10, y)
                    if label_pt:
                        text = f"{int(y)}"
                        tw   = fm.horizontalAdvance(text)
                        th   = fm.height()
                        painter.fillRect(
                            QRectF(label_pt.x() - tw - 2, label_pt.y() - th + 2,
                                tw + 4, th),
                            QColor(255, 255, 255, 200)
                        )
                        painter.setPen(QPen(QColor(0, 0, 0)))
                        painter.drawText(
                            QPointF(label_pt.x() - tw, label_pt.y()),
                            text
                        )
            y += spacing

        # ── Origin marker (0,0) sits just outside the ID 0 corner in mat space ──
        origin = self._mm_to_scene(-12, -10)
        if origin:
            font_origin = QFont("Helvetica", 11, QFont.Weight.Bold)
            painter.setFont(font_origin)
            fm2 = painter.fontMetrics()
            text = "(0,0)"
            tw   = fm2.horizontalAdvance(text)
            th   = fm2.height()
            painter.fillRect(
                QRectF(origin.x() - tw - 2, origin.y() - th + 2,
                    tw + 4, th),
                QColor(255, 255, 255, 220)
            )
            painter.setPen(QPen(QColor(0, 80, 220)))
            painter.drawText(QPointF(origin.x() - tw, origin.y()), text)






    def _mm_to_scene(self, x_mm: float, y_mm: float):
        """Convert mm (mat frame) → scene pixel via inverse homography + scale."""
        if self._H_inv is None:
            return None
        pt     = np.array([[[x_mm, y_mm]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H_inv)
        cam_x  = float(result[0][0][0])
        cam_y  = float(result[0][0][1])
        sx     = cam_x * self._scale_x + self._offset_x
        sy     = cam_y * self._scale_y + self._offset_y
        return QPointF(sx, sy)


# ── Scene manager ─────────────────────────────────────────────────────────────

class SceneManager:
    """Handles scene setup and image loading"""

    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self._pixmap_item: QGraphicsPixmapItem = None
        self._image_offset_x: float = 0.0
        self._image_offset_y: float = 0.0
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

        # Fixed pixel grid (legacy, kept for toggle_grid compat)
        self._grid_overlay: GridOverlay = None
        self._grid_visible: bool = False

        # Perspective-correct homography grid
        self._hgrid: HomographyGridOverlay = None

    def setup_scene(self) -> None:
        self.scene.setSceneRect(
            0, 0,
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT
        )

    def clear(self) -> None:
        self.scene.clear()
        self._pixmap_item = None
        self._grid_overlay = None
        self._hgrid = None

    def load_body_diagram(self) -> bool:
        image_path = settings.BODY_DIAGRAM_IMAGE

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            self._load_placeholder("Image Not Found\n(Using Placeholder)")
            return False

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Warning: Failed to load image from {image_path}")
            self._load_placeholder("Failed to Load Image")
            return False

        self._display_pixmap(pixmap)
        return True

    def load_captured_image(self, cv_image) -> None:
        self.scene.clear()
        self._pixmap_item = None
        self._grid_overlay = None
        self._hgrid = None

        image_rgb      = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        q_image        = QImage(image_rgb.data, width, height,
                                bytes_per_line, QImage.Format.Format_RGB888)
        pixmap         = QPixmap.fromImage(q_image)

        self._display_pixmap(pixmap)
        print(f"✓ Captured image loaded: {width}×{height}")

    def update_live_frame(self, cv_image) -> None:
        """
        Update background image WITHOUT clearing electrodes or calibration.
        Stores scale factors for homography grid conversion.
        """
        image_rgb      = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        q_image        = QImage(image_rgb.data, width, height,
                                bytes_per_line, QImage.Format.Format_RGB888)
        pixmap         = QPixmap.fromImage(q_image)

        scaled = pixmap.scaled(
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )

        x_offset = (settings.BODY_DIAGRAM_WIDTH  - scaled.width())  / 2
        y_offset = (settings.BODY_DIAGRAM_HEIGHT - scaled.height()) / 2

        # Store scale factors for grid conversion
        self._scale_x     = scaled.width()  / width
        self._scale_y     = scaled.height() / height
        self._image_offset_x = x_offset
        self._image_offset_y = y_offset

        if self._pixmap_item is not None:
            self._pixmap_item.setPixmap(scaled)
            self._pixmap_item.setPos(x_offset, y_offset)
        else:
            self._pixmap_item = self.scene.addPixmap(scaled)
            self._pixmap_item.setPos(x_offset, y_offset)
            self._pixmap_item.setZValue(-1000)
            self._add_border()
            self.setup_scene()

        self._ensure_hgrid()

    def update_aruco_grid(self, aruco_state: dict) -> None:
        """
        Update the perspective-correct grid from the ArUco homography.
        Called every frame when mat is calibrated.
        """
        # Show grid as soon as mat is calibrated — don't wait for full ready
        mat_ready = aruco_state.get("mat_ready") or aruco_state.get("ready")
        if not mat_ready:
            if self._hgrid is not None:
                self._hgrid.setVisible(False)
            return

        H = aruco_state.get("homography")
        if H is None:
            return

        self._ensure_hgrid()
        self._hgrid.update_transform(
            H,
            self._scale_x,
            self._scale_y,
            self._image_offset_x,
            self._image_offset_y,
        )
        self._hgrid.setVisible(True)


    def _ensure_hgrid(self) -> None:
        """Create homography grid overlay if it doesn't exist yet."""
        if self._hgrid is None:
            self._hgrid = HomographyGridOverlay()
            self.scene.addItem(self._hgrid)

    def _display_pixmap(self, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        x_offset = (settings.BODY_DIAGRAM_WIDTH  - scaled.width())  / 2
        y_offset = (settings.BODY_DIAGRAM_HEIGHT - scaled.height()) / 2

        self._pixmap_item = self.scene.addPixmap(scaled)
        self._pixmap_item.setPos(x_offset, y_offset)
        self._pixmap_item.setZValue(-1000)

        self._image_offset_x = x_offset
        self._image_offset_y = y_offset
        self._scale_x = scaled.width()  / pixmap.width()  if pixmap.width()  else 1.0
        self._scale_y = scaled.height() / pixmap.height() if pixmap.height() else 1.0

        self._grid_overlay = None
        self._ensure_grid(x_offset, y_offset, scaled.width(), scaled.height())

        self._add_border()
        self.setup_scene()

        print(f"Loaded image: {scaled.width()}×{scaled.height()} pixels")

    def _add_border(self) -> None:
        border = QGraphicsRectItem(
            0, 0,
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT
        )
        border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        border.setPen(QPen(Qt.GlobalColor.black, 2))
        self.scene.addItem(border)

    # ── Legacy fixed grid ─────────────────────────────────────────────────────

    def _ensure_grid(self, x: float, y: float, w: float, h: float) -> None:
        if self._grid_overlay is None:
            self._grid_overlay = GridOverlay(w, h)
            self.scene.addItem(self._grid_overlay)
        else:
            self._grid_overlay.update_size(w, h)
        self._grid_overlay.setPos(x, y)
        self._grid_overlay.setVisible(self._grid_visible)

    def toggle_grid(self, visible: bool) -> None:
        self._grid_visible = visible
        if self._grid_overlay is not None:
            self._grid_overlay.setVisible(visible)

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _load_placeholder(self, message: str) -> None:
        rect = QGraphicsRectItem(
            0, 0,
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT
        )
        rect.setBrush(QBrush(Qt.GlobalColor.white))
        rect.setPen(QPen(Qt.GlobalColor.black, 2))
        self.scene.addItem(rect)

        text = QGraphicsTextItem(message)
        text.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        text.setDefaultTextColor(Qt.GlobalColor.red)
        text.setPos(100, 280)
        self.scene.addItem(text)

        self.setup_scene()