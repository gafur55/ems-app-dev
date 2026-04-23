from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal


class CaptureControls(QWidget):
    """
    Widget containing the Place Electrodes button.

    Signals:
        place_electrodes_requested: User wants to calibrate and place electrodes
    """

    place_electrodes_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.place_btn = QPushButton("Place Electrodes")
        self.place_btn.clicked.connect(self._on_place_clicked)
        self.place_btn.setEnabled(False)
        self.place_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #FFA726; }
            QPushButton:disabled { background-color: #CCCCCC; }
        """)
        layout.addWidget(self.place_btn)

    def set_camera_active(self, active: bool) -> None:
        self.place_btn.setEnabled(active)

    def reset(self) -> None:
        pass

    def _on_place_clicked(self) -> None:
        print("\n Place Electrodes requested")
        self.place_electrodes_requested.emit()