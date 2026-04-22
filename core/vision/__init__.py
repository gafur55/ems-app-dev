"""
Vision processing module for coordinate mapping, calibration, and hand tracking
"""

from .anatomical_mapper import AnatomicalCoordinateMapper


try:
    from .hand_tracking import (
        HandDetector,
        HandDetectorSimple,
        AngleCalculator,
        MovementAnalyzer,
        MovementResult,
        HandLandmark,
        MEDIAPIPE_AVAILABLE,
    )
    HAND_TRACKING_AVAILABLE = True
except ImportError:
    HAND_TRACKING_AVAILABLE = False

__all__ = [
    'AnatomicalCoordinateMapper',
    'HAND_TRACKING_AVAILABLE',
]

# Add hand tracking exports if available
if HAND_TRACKING_AVAILABLE:
    __all__.extend([
        'HandDetector',
        'HandDetectorSimple', 
        'AngleCalculator',
        'MovementAnalyzer',
        'MovementResult',
        'HandLandmark',
        'MEDIAPIPE_AVAILABLE',
    ])