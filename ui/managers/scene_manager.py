"""
Manages the graphics scene: setup, image loading, placeholders, live frame updates.
"""

from PyQt6.QtWidgets import QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem, QGraphicsPixmapItem
from PyQt6.QtGui import QBrush, QPen, QFont, QPixmap, QImage
from PyQt6.QtCore import Qt
from config import settings
from ui.managers.grid_overlay import GridOverlay
import os


class SceneManager:
    """Handles scene setup and image loading"""
    
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self._pixmap_item: QGraphicsPixmapItem = None  # reference to background image
        self._image_offset_x: float = 0.0
        self._image_offset_y: float = 0.0

        # Grid overlay — created lazily when the first image arrives
        self._grid_overlay: GridOverlay = None
        self._grid_visible: bool = False
    
    def setup_scene(self) -> None:
        """Initialize scene with default boundaries"""
        self.scene.setSceneRect(
            0, 0,
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT
        )
    
    def clear(self) -> None:
        """Clear all items from scene"""
        self.scene.clear()
        self._pixmap_item = None
        self._grid_overlay = None   # scene.clear() destroys the item
    
    def load_body_diagram(self) -> bool:
        """
        Load the forearm diagram image from settings.
        
        Returns:
            True if image loaded successfully, False if placeholder used
        """
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
        """
        Load a captured OpenCV image (BGR format) into the scene.
        Replaces any existing content.
        
        Args:
            cv_image: OpenCV image in BGR format
        """
        import cv2
        
        self.scene.clear()
        self._pixmap_item = None
        self._grid_overlay = None   # scene.clear() destroys it
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        
        # Convert to QImage then QPixmap
        bytes_per_line = channels * width
        q_image = QImage(
            image_rgb.data, width, height,
            bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        
        self._display_pixmap(pixmap)
        print(f"✓ Captured image loaded: {width}×{height}")

    def update_live_frame(self, cv_image) -> None:
        """
        Update the background image WITHOUT clearing electrodes or calibration markers.
        
        This swaps just the pixmap data on the existing background item.
        If no background item exists yet, creates one at the bottom of the z-order.
        
        Args:
            cv_image: OpenCV image in BGR format
        """
        import cv2
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        
        bytes_per_line = channels * width
        q_image = QImage(
            image_rgb.data, width, height,
            bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit
        scaled = pixmap.scaled(
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation  # Fast for live updates
        )
        
        x_offset = (settings.BODY_DIAGRAM_WIDTH - scaled.width()) / 2
        y_offset = (settings.BODY_DIAGRAM_HEIGHT - scaled.height()) / 2
        
        if self._pixmap_item is not None:
            # Just swap the pixmap — no scene clearing
            self._pixmap_item.setPixmap(scaled)
            self._pixmap_item.setPos(x_offset, y_offset)
        else:
            # First frame — create the item at the bottom of z-order
            self._pixmap_item = self.scene.addPixmap(scaled)
            self._pixmap_item.setPos(x_offset, y_offset)
            self._pixmap_item.setZValue(-1000)  # behind everything else
            self._add_border()
            self.setup_scene()
        
        self._image_offset_x = x_offset
        self._image_offset_y = y_offset

        # Keep grid in sync with the (possibly resized) image
        self._ensure_grid(x_offset, y_offset, scaled.width(), scaled.height())

    def _display_pixmap(self, pixmap: QPixmap) -> None:
        """
        Scale and center a pixmap in the scene.
        
        Args:
            pixmap: QPixmap to display
        """
        scaled = pixmap.scaled(
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center in scene
        x_offset = (settings.BODY_DIAGRAM_WIDTH - scaled.width()) / 2
        y_offset = (settings.BODY_DIAGRAM_HEIGHT - scaled.height()) / 2
        
        self._pixmap_item = self.scene.addPixmap(scaled)
        self._pixmap_item.setPos(x_offset, y_offset)
        self._pixmap_item.setZValue(-1000)
        
        self._image_offset_x = x_offset
        self._image_offset_y = y_offset

        # Rebuild grid (scene was cleared before this call)
        self._grid_overlay = None
        self._ensure_grid(x_offset, y_offset, scaled.width(), scaled.height())
        
        self._add_border()
        self.setup_scene()
        
        print(f"Loaded image: {scaled.width()}×{scaled.height()} pixels")
    
    def _add_border(self) -> None:
        """Add border rectangle around canvas"""
        border = QGraphicsRectItem(
            0, 0,
            settings.BODY_DIAGRAM_WIDTH,
            settings.BODY_DIAGRAM_HEIGHT
        )
        border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        border.setPen(QPen(Qt.GlobalColor.black, 2))
        self.scene.addItem(border)

    # ── Grid overlay ──────────────────────────────────────────────────────────

    def _ensure_grid(self, x: float, y: float, w: float, h: float) -> None:
        """
        Create the grid overlay if it doesn't exist yet, or reposition/resize
        it to match the current image bounds.
        """
        if self._grid_overlay is None:
            self._grid_overlay = GridOverlay(w, h)
            self.scene.addItem(self._grid_overlay)
        else:
            self._grid_overlay.update_size(w, h)

        self._grid_overlay.setPos(x, y)
        self._grid_overlay.setVisible(self._grid_visible)

    def toggle_grid(self, visible: bool) -> None:
        """Show or hide the grid overlay."""
        self._grid_visible = visible
        if self._grid_overlay is not None:
            self._grid_overlay.setVisible(visible)

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _load_placeholder(self, message: str) -> None:
        """
        Display placeholder rectangle with error message.
        
        Args:
            message: Text to display in placeholder
        """
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