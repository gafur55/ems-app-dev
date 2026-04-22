"""
Data models for EMS sessions, electrodes, and parameters
"""

from datetime import datetime
import math


class Session:
    """Represents an EMS session"""
    
    def __init__(self, participant_id=None, device_name="p24"):
        self.session_id = None  # Will be set by database
        self.participant_id = participant_id
        self.device_name = device_name
        self.timestamp = datetime.now()
        
        # Personal parameters
        self.age = None
        self.arm_width = None
        self.arm_length = None
        self.pain_threshold = None
        self.experience_level = None
        self.skin_impedance = None
        
        # NEW: Forearm capture data
        self.forearm_image_path = None        # Path to saved image
        self.forearm_image_width_px = None    # Image width in pixels
        self.forearm_image_height_px = None   # Image height in pixels
        self.forearm_captured = False         # Was image captured?
        
        # NEW: Real-world measurements
        self.forearm_length_measured_cm = None  # Auto-measured length
        self.forearm_width_measured_cm = None   # Auto-measured width
        
        # NEW: Coordinate system calibration
        self.calibration_factor = None        # Pixels per cm
        self.origin_pixel_x = None           # Wrist position in image (x)
        self.origin_pixel_y = None           # Wrist position in image (y)
        
        # NEW: Target gesture
        self.target_gesture = None           # "finger_flexion", etc.
        
        # Session data
        self.electrodes = []
        self.stimulation_events = []


class Electrode:
    """Represents an electrode placement with dual coordinates"""
    
    def __init__(self, channel, pixel_x, pixel_y):
        self.electrode_id = None  # Will be set by database
        self.channel = channel    # 0-7
        self.timestamp = datetime.now()
        self.active = True
        
        # PIXEL COORDINATES (as placed in image)
        self.pixel_x = pixel_x              # X position in captured image
        self.pixel_y = pixel_y              # Y position in captured image
        
        # REAL-WORLD COORDINATES (anatomical position)
        self.real_x_cm = None               # Distance from wrist along forearm
        self.real_y_cm = None               # Distance from centerline across forearm
        
        # NORMALIZED ANATOMICAL COORDINATES (NEW)
        self.normalized_longitudinal = None  # s (0 = wrist, 1 = elbow)
        self.normalized_lateral = None       # t (normalized by width)

        # ANATOMICAL CONTEXT
        self.anatomical_zone = None         # "flexor_digitorum_superficialis", etc.
        self.distance_from_wrist_cm = None  # Direct distance from wrist landmark
        self.distance_from_elbow_cm = None  # Direct distance from elbow landmark
        
        # PLACEMENT METADATA
        self.placement_method = "manual"    # "manual", "marker_guided", "auto_suggested"
        self.placement_confidence = None    # If auto-placed, confidence score
    
    def calculate_real_coordinates(self, origin_px, origin_py, calibration_factor):
        """
        Calculate real-world coordinates from pixel position.
        
        Args:
            origin_px: Wrist X position in pixels (origin point)
            origin_py: Wrist Y position in pixels (origin point)
            calibration_factor: Pixels per centimeter
        
        Returns:
            Tuple (real_x_cm, real_y_cm)
        """
        if calibration_factor is None or calibration_factor == 0:
            return None, None
        
        # Calculate offset from origin (wrist)
        delta_x_px = self.pixel_x - origin_px
        delta_y_px = self.pixel_y - origin_py
        
        # Convert to centimeters
        self.real_x_cm = delta_x_px / calibration_factor
        self.real_y_cm = delta_y_px / calibration_factor
        
        # Calculate direct distance from wrist
        self.distance_from_wrist_cm = math.sqrt(
            self.real_x_cm**2 + self.real_y_cm**2
        )
        
        return self.real_x_cm, self.real_y_cm
    
    def to_dict(self):
        """Export electrode data as dictionary"""
        return {
            "electrode_id": self.electrode_id,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat(),
            "active": self.active,
            # Pixel coordinates
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            # Real-world coordinates
            "real_x_cm": self.real_x_cm,
            "real_y_cm": self.real_y_cm,
            "distance_from_wrist_cm": self.distance_from_wrist_cm,
            "distance_from_elbow_cm": self.distance_from_elbow_cm,
            # Context
            "anatomical_zone": self.anatomical_zone,
            "placement_method": self.placement_method,
        }


class StimulationEvent:
    """Represents a single stimulation event"""
    
    def __init__(self, channel, intensity, pulse_width):
        self.event_id = None
        self.channel = channel
        self.intensity_ma = intensity
        self.pulse_width_us = pulse_width
        self.pulse_count = None
        self.delay_ms = None
        self.timestamp = datetime.now()
        self.device_name = None
        
        # NEW: Link to electrode positions used
        self.electrode_positions = []  # List of {channel, real_x, real_y}
        self.success = None           # True/False/None
        self.notes = None             # Researcher observations