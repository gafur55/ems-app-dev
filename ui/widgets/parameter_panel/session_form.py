"""
Session form widget for session-level parameters only.

Personal info (age, arm measurements, etc.) is now handled
by the sign-in dialog and stored in the database.

This form only contains:
- Device selection
"""

from PyQt6.QtWidgets import (QWidget, QGridLayout, QLabel,
                              QComboBox, QGroupBox, QVBoxLayout)
from config import settings


class SessionForm(QWidget):
    """
    Widget containing session parameter input fields.

    Only device-level settings — no personal info.
    """

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the form layout"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Create group box
        group = QGroupBox("Session Settings")
        group_layout = QGridLayout()
        group.setLayout(group_layout)

        # Device
        group_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(list(settings.SUPPORTED_DEVICES.keys()))
        self.device_combo.setCurrentText(settings.DEFAULT_DEVICE)
        group_layout.addWidget(self.device_combo, 0, 1)

        layout.addWidget(group)

    def get_parameters(self) -> dict:
        """
        Get session parameters.

        Returns:
            dict with device_name
        """
        return {
            "device_name": self.device_combo.currentText(),
        }

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable inputs."""
        self.device_combo.setEnabled(enabled)
