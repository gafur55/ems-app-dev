"""
Parameter panel - coordinates session form and capture controls.

Personal info is no longer here — it's handled by the sign-in dialog.
This panel only manages: device selection, capture, calibration, start/stop.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel)
from PyQt6.QtCore import pyqtSignal

from .session_form import SessionForm
from .capture_controls import CaptureControls


class ParameterPanel(QWidget):
    """
    Main parameter panel that coordinates session inputs and capture controls.

    Signals:
        session_started: Emitted with parameters dict when session starts
        session_stopped: Emitted when session stops
        calibration_requested: Emitted when calibration button clicked
    """

    # Signals
    session_started = pyqtSignal(dict)
    session_stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.session_active = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the panel layout"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Session form (device selection only)
        self.session_form = SessionForm()
        layout.addWidget(self.session_form)

        # Capture controls (Pause/Resume, Calibrate, Clear)
        self.capture_controls = CaptureControls()
        layout.addWidget(self.capture_controls)

        # Session control buttons
        self._setup_session_buttons(layout)

        # Status label
        self.status_label = QLabel("Status: No active session")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)

    def _setup_session_buttons(self, parent_layout: QVBoxLayout) -> None:
        """Set up start/stop session buttons"""
        button_layout = QHBoxLayout()

        # Start Session button
        self.start_button = QPushButton("Start Session")
        self.start_button.clicked.connect(self._on_start_session)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #00BF00;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #00D000; }
            QPushButton:disabled { background-color: #CCCCCC; }
        """)

        # Stop Session button
        self.stop_button = QPushButton("Stop Session")
        self.stop_button.clicked.connect(self._on_stop_session)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #FF6347; }
            QPushButton:disabled { background-color: #CCCCCC; }
        """)

        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()

        parent_layout.addLayout(button_layout)

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_start_session(self) -> None:
        """Handle Start Session button click"""
        params = self.session_form.get_parameters()

        # Update UI state
        self.session_active = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"Status: Session active (Device: {params['device_name']})")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        # Disable inputs during session
        self.session_form.set_enabled(False)

        # Emit signal
        self.session_started.emit(params)
        print(f"Session started with device: {params['device_name']}")

    def _on_stop_session(self) -> None:
        """Handle Stop Session button click"""
        # Update UI state
        self.session_active = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Session stopped")
        self.status_label.setStyleSheet("color: orange; font-style: italic;")

        # Re-enable inputs
        self.session_form.set_enabled(True)
        self.capture_controls.reset()

        # Emit signal
        self.session_stopped.emit()
        print("Session stopped")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_parameters(self) -> dict:
        """Get all parameter values"""
        return self.session_form.get_parameters()

    def set_inputs_enabled(self, enabled: bool) -> None:
        """Enable/disable all input fields"""
        self.session_form.set_enabled(enabled)