"""
Stimulation control panel for EMS parameters and triggering
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel,
                              QLineEdit, QComboBox, QPushButton, QGroupBox)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from config import settings


class StimulationPanel(QWidget):
    """
    Widget for controlling stimulation parameters.
    Emits signal when stimulate button is clicked.
    """
    
    # Signal emitted when "Stimulate" is clicked with parameters
    stimulate_requested = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    

    def setup_ui(self):
        """Set up the stimulation control interface"""
        device_info = settings.SUPPORTED_DEVICES[settings.DEFAULT_DEVICE]
        max_intensity = device_info['max_intensity']
        max_pulse_width = device_info['max_pulse_width']

        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create group box for stimulation controls
        group = QGroupBox("Stimulation Controls")
        group_layout = QGridLayout()
        group.setLayout(group_layout)
        
        # Row 0: Channel selector
        group_layout.addWidget(QLabel("Channel:"), 0, 0)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([str(i) for i in range(settings.MAX_CHANNELS)])
        self.channel_combo.setCurrentText("6")    
        group_layout.addWidget(self.channel_combo, 0, 1)
        
        # Row 1: Intensity
        group_layout.addWidget(QLabel("Intensity (mA):"), 1, 0)
        self.intensity_input = QLineEdit()
        # Get max intensity from the default device settings
        device_info = settings.SUPPORTED_DEVICES[settings.DEFAULT_DEVICE]
        self.intensity_input.setPlaceholderText(f"0-{max_intensity}")
        self.intensity_input.setText(str(settings.DEFAULT_INTENSITY))
        self.intensity_input.setValidator(QIntValidator(
            settings.MIN_INTENSITY, 
            settings.MAX_INTENSITY
        ))
        group_layout.addWidget(self.intensity_input, 1, 1)
        
        # Row 2: Pulse Width
        group_layout.addWidget(QLabel("Pulse Width (μs):"), 2, 0)
        self.pulse_width_input = QLineEdit()
        self.pulse_width_input.setPlaceholderText(f"0-{max_pulse_width}")
        self.pulse_width_input.setText(str(settings.DEFAULT_PULSE_WIDTH))
        self.pulse_width_input.setValidator(QIntValidator(
            settings.MIN_PULSE_WIDTH,
            settings.MAX_PULSE_WIDTH
        ))
        group_layout.addWidget(self.pulse_width_input, 2, 1)
        
        # Row 3: Pulse Count
        group_layout.addWidget(QLabel("Pulse Count:"), 3, 0)
        self.pulse_count_input = QLineEdit()
        self.pulse_count_input.setPlaceholderText("e.g., 10")
        self.pulse_count_input.setText("10")
        self.pulse_count_input.setValidator(QIntValidator(1, 1000))
        group_layout.addWidget(self.pulse_count_input, 3, 1)
        
        # Row 4: Delay
        group_layout.addWidget(QLabel("Delay (ms):"), 4, 0)
        self.delay_input = QLineEdit()
        self.delay_input.setPlaceholderText("e.g., 10")
        self.delay_input.setText("10")
        self.delay_input.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        group_layout.addWidget(self.delay_input, 4, 1)
        
        layout.addWidget(group)
        
        # Stimulate button
        self.stimulate_button = QPushButton("STIMULATE")
        self.stimulate_button.clicked.connect(self.on_stimulate)
        self.stimulate_button.setStyleSheet("""
            QPushButton {
                background-color: #800000;
                color: white;
                font-weight: bold;
                padding: 15px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #FF6347;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        layout.addWidget(self.stimulate_button)
        
        # Status label (only shown for validation errors)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        
        # Initially disabled until session starts
        self.setEnabled(False)

    
    def on_stimulate(self):
        """Handle Stimulate button click"""
        params = self.get_parameters()
        
        # Validate parameters before emitting
        if not self.validate_parameters(params):
            return  # Don't stimulate if validation fails
        
        # Emit signal with parameters
        self.stimulate_requested.emit(params)

        print(f"Stimulation requested: Channel {params['channel']}, "
              f"{params['intensity']}mA, {params['pulse_width']}μs")
    

    def get_parameters(self):
        """
        Get all stimulation parameters.
        
        Returns:
            dict: Stimulation parameters
        """
        def safe_int(value, default=0):
            try:
                return int(value) if value.strip() else default
            except ValueError:
                return default
        
        def safe_float(value, default=0.0):
            try:
                return float(value) if value.strip() else default
            except ValueError:
                return default
        
        return {
            "channel": int(self.channel_combo.currentText()),
            "intensity": safe_int(self.intensity_input.text(), settings.DEFAULT_INTENSITY),
            "pulse_width": safe_int(self.pulse_width_input.text(), settings.DEFAULT_PULSE_WIDTH),
            "pulse_count": safe_int(self.pulse_count_input.text(), 10),
            "delay": safe_float(self.delay_input.text(), 10.0)
        }
    

    def validate_parameters(self, params):
        """
        Validate stimulation parameters before allowing stimulation.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Get device limits
        device_info = settings.SUPPORTED_DEVICES[settings.DEFAULT_DEVICE]
        max_intensity = device_info['max_intensity']
        max_pulse_width = device_info['max_pulse_width']
        
        # Check intensity
        if params['intensity'] < 0 or params['intensity'] > max_intensity:
            self.set_status(f"Error: Intensity must be 0-{max_intensity} mA", "red")
            return False
        
        # Check pulse width
        if params['pulse_width'] < 0 or params['pulse_width'] > max_pulse_width:
            self.set_status(f"Error: Pulse width must be 0-{max_pulse_width} μs", "red")
            return False
        
        # Check pulse count
        if params['pulse_count'] < 1:
            self.set_status("Error: Pulse count must be at least 1", "red")
            return False
        
        # Check delay
        if params['delay'] < 0:
            self.set_status("Error: Delay cannot be negative", "red")
            return False
        
        # All valid
        self.set_status("Parameters valid", "green")
        return True
    


    def set_status(self, message, color="gray"):
        """
        Update status label (shows for validation errors).
        """
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; font-style: italic;")
        self.status_label.setVisible(bool(message))


        # ── Hotkey helpers ────────────────────────────────────────────────────

    def increment_intensity(self, step: int = 1) -> None:
        """Bump intensity by `step` mA (clamped to device max)."""
        device_info = settings.SUPPORTED_DEVICES[settings.DEFAULT_DEVICE]
        max_intensity = device_info["max_intensity"]
        try:
            current = int(self.intensity_input.text())
        except ValueError:
            current = settings.DEFAULT_INTENSITY
        new_val = max(settings.MIN_INTENSITY, min(max_intensity, current + step))
        self.intensity_input.setText(str(new_val))
        self.set_status(f"Intensity: {new_val} mA", "lightblue")

    def increment_pulse_width(self, step: int = 10) -> None:
        """Bump pulse width by `step` μs (clamped to device max)."""
        device_info = settings.SUPPORTED_DEVICES[settings.DEFAULT_DEVICE]
        max_pw = device_info["max_pulse_width"]
        try:
            current = int(self.pulse_width_input.text())
        except ValueError:
            current = settings.DEFAULT_PULSE_WIDTH
        new_val = max(settings.MIN_PULSE_WIDTH, min(max_pw, current + step))
        self.pulse_width_input.setText(str(new_val))
        self.set_status(f"Pulse width: {new_val} μs", "lightblue")

    def increment_pulse_count(self, step: int = 5) -> None:
        """Bump pulse count by `step` (min 1)."""
        try:
            current = int(self.pulse_count_input.text())
        except ValueError:
            current = 10
        new_val = max(1, current + step)
        self.pulse_count_input.setText(str(new_val))
        self.set_status(f"Pulse count: {new_val}", "lightblue")

    def reset_to_defaults(self) -> None:
        """Reset all parameters to defaults (R key)."""
        self.intensity_input.setText(str(settings.DEFAULT_INTENSITY))
        self.pulse_width_input.setText(str(settings.DEFAULT_PULSE_WIDTH))
        self.pulse_count_input.setText("10")
        self.delay_input.setText("10")
        self.channel_combo.setCurrentText("6")
        self.set_status("Parameters reset to defaults", "lightgreen")

    def trigger_stimulate(self) -> None:
        """Programmatic stim trigger (Space key)."""
        if self.isEnabled():
            self.on_stimulate()