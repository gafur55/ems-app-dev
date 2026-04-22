"""
Grid overlay drawn directly on top of the forearm image.

Implemented as a single custom QGraphicsItem so the entire grid is painted
in one pass — no per-line scene items, no heap of QGraphicsLineItems.

Z-value: -999  (above the background image at -1000, below electrodes / calibration at 0+)
"""

from PyQt6.QtWidgets import QGraphicsItem
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import QRectF, Qt


# ── tuneable constants ───────────────────────────────────────────────────────
GRID_CELL_SIZE_PX   = 15          # pixels between grid lines (scene coordinates)
GRID_LINE_COLOR     = QColor(0, 0, 0, 250)   # black, ~22 % opacity
GRID_LINE_WIDTH     = 0.6         # cosmetic (sub-pixel) line width
GRID_Z_VALUE        = -999        # just above the background image
# ────────────────────────────────────────────────────────────────────────────


class GridOverlay(QGraphicsItem):
    """
    Lightweight grid drawn over the forearm image.

    Usage
    -----
    overlay = GridOverlay(width, height)
    scene.addItem(overlay)
    overlay.setPos(x_offset, y_offset)
    overlay.setZValue(GRID_Z_VALUE)

    # resize when the image changes:
    overlay.update_size(new_w, new_h)
    overlay.setPos(new_x, new_y)
    """

    def __init__(
        self,
        width: float,
        height: float,
        cell_size: int = GRID_CELL_SIZE_PX,
        color: QColor = GRID_LINE_COLOR,
        line_width: float = GRID_LINE_WIDTH,
    ):
        super().__init__()
        self._width     = width
        self._height    = height
        self._cell_size = cell_size
        self._color     = color
        self._line_width = line_width

        self._pen = QPen(self._color, self._line_width)
        self._pen.setCosmetic(True)          # stays thin regardless of zoom

        # Don't interfere with mouse events meant for electrodes / calibration
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable,    False)

        self.setZValue(GRID_Z_VALUE)

    # ── QGraphicsItem interface ──────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setPen(self._pen)

        cs = self._cell_size

        # Vertical lines
        x = 0.0
        while x <= self._width:
            painter.drawLine(int(x), 0, int(x), int(self._height))
            x += cs

        # Horizontal lines
        y = 0.0
        while y <= self._height:
            painter.drawLine(0, int(y), int(self._width), int(y))
            y += cs

    # ── Public helpers ───────────────────────────────────────────────────────

    def update_size(self, width: float, height: float) -> None:
        """Call when the image is resized so the grid covers it exactly."""
        self.prepareGeometryChange()
        self._width  = width
        self._height = height
        self.update()

    def set_cell_size(self, cell_size: int) -> None:
        """Change grid spacing at runtime."""
        self._cell_size = cell_size
        self.update()

    def set_visible(self, visible: bool) -> None:
        """Show / hide the grid without removing it from the scene."""
        self.setVisible(visible)