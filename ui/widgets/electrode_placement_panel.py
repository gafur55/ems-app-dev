"""
Electrode placement panel widget.

Provides a 'Confirm Electrode Placement' button that gates
stimulation — user must place electrodes and confirm before
the STIMULATE button becomes active.

Emits:
    electrodes_confirmed: Signal when user confirms placement
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout,
    QPushButton, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt


class ElectrodePlacementPanel(QWidget):
    """
    Panel for confirming electrode placement.
    Minimal — just the confirm button. Status shown via main status label.
    """

    electrodes_confirmed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.setEnabled(False)

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        self.setLayout(layout)

        # Confirm button only
        self.confirm_button = QPushButton("Confirm Electrode Placement")
        self.confirm_button.setMinimumHeight(40)
        self.confirm_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_button_style_waiting()
        self.confirm_button.clicked.connect(self._on_confirm_clicked)
        layout.addWidget(self.confirm_button)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_confirmed(self, electrode_count: int) -> None:
        self.confirm_button.setText(f"Electrodes Placed ({electrode_count})")
        self.confirm_button.setEnabled(False)
        self._set_button_style_confirmed()

    def set_error(self, message: str) -> None:
        """Briefly flash error on button text."""
        self.confirm_button.setText(f"{message}")

    def reset(self) -> None:
        self.confirm_button.setText("Confirm Electrode Placement")
        self.confirm_button.setEnabled(True)
        self._set_button_style_waiting()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_confirm_clicked(self):
        self.electrodes_confirmed.emit()

    def _set_button_style_waiting(self):
        self.confirm_button.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:disabled {
                background-color: #6b7280;
                color: #9ca3af;
            }
        """)

    def _set_button_style_confirmed(self):
        self.confirm_button.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:disabled {
                background-color: #16a34a;
                color: white;
            }
        """)