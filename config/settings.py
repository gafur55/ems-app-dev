"""
Configuration settings for EMS GUI application
"""

# Application Settings
APP_NAME = "EMS Control GUI"
APP_VERSION = "0.1.0"

# Window Settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700

# Body Diagram Settings
BODY_DIAGRAM_WIDTH = 1000
BODY_DIAGRAM_HEIGHT = 700

# UI Colors
BACKGROUND_COLOR = "white"
CANVAS_BORDER_COLOR = "black"

# Body Diagram Image
BODY_DIAGRAM_IMAGE = "assets/forearm_palmar.png"  # Update extension if needed

# Supported EMS Devices
SUPPORTED_DEVICES = {
    "p24": {
        "name": "P24",
        "max_channels": 8,
        "max_intensity": 100,
        "max_pulse_width": 500,
        "description": "Primary lab device"
    },
    # Add other devices here as needed
    # "device_name": {...},
}

# Default device
DEFAULT_DEVICE = "p24"

MAX_CHANNELS = 8  # Device supports channels 0-7
ELECTRODES_PER_CHANNEL = 2  # Each channel needs 2 electrodes (pair)
MAX_ELECTRODES = MAX_CHANNELS * ELECTRODES_PER_CHANNEL  # Total: 16

# Electrode Visual Settings
ELECTRODE_RADIUS = 15  # pixels
ELECTRODE_BORDER_WIDTH = 2  # pixels

# Electrode Colors (RGB)
ELECTRODE_COLOR = (0, 191, 255)  # Red
ELECTRODE_ACTIVE_COLOR = (0, 255, 0)  # Green
ELECTRODE_INACTIVE_COLOR = (128, 128, 128)  # Gray

# Safety Settings
DEFAULT_INTENSITY = 5  # mA (safe starting point)
DEFAULT_PULSE_WIDTH = 250  # microseconds

MIN_INTENSITY = 0  # mA
MAX_INTENSITY = 100  # mA
MIN_PULSE_WIDTH = 0  # microseconds
MAX_PULSE_WIDTH = 500  # microseconds


MARKER_SIZE_MM = 16.0  # physical size of printed ArUco markers in mm


# Physical mat dimensions — measure your actual printed mat
# Markers are placed at exact corners of this rectangle
MAT_WIDTH_MM  = 199.0   # left-right distance (ID0 → ID1)
MAT_HEIGHT_MM = 380.0   # top-bottom distance (ID0 → ID3)

# ArUco marker ID assignments
# Mat corners (fixed to surface, never move)
MAT_IDS =       [0, 2, 3, 4]   # TL, TR, BR, BL

# Wrist marker (placed on participant's wrist)
WRIST_ID      = 1

# Electrode markers (placed on electrode pads)
ELECTRODE_IDS = [5, 6]
# Add more IDs here if tracking more than 2 electrodes per session
# e.g. ELECTRODE_IDS = [5, 6, 7, 8] for 4 electrodes

ELBOW_ID = 7
WRIST_RADIAL_ID = 8  