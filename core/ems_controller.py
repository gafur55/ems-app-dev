"""
EMS Device Controller
Wrapper for the EMS library to handle device connection and stimulation
"""

import sys
import os

# Add the EMS library path to system path
EMS_LIB_PATH = "/Users/gafurmammadov/Documents/Uchicago_classes/practicum/ems_lib/ems-main/ems"
if EMS_LIB_PATH not in sys.path:
    sys.path.insert(0, EMS_LIB_PATH)

try:
    # Import directly from the EMS core.py file
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ems_core",
        "/Users/gafurmammadov/Documents/Uchicago_classes/practicum/ems_lib/ems-main/ems/core.py"
    )
    ems_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ems_module)

    EMS = ems_module.EMS
    EMS_AVAILABLE = True
    print("✓ EMS library loaded successfully")

except Exception as e:
    print(f"Warning: Could not import EMS library: {e}")
    EMS_AVAILABLE = False


class EMSController:
    """
    Controller for EMS device.
    Handles connection, stimulation, and error handling.
    """
    
    def __init__(self, device_name="p24", debug=True):
        """
        Initialize EMS controller.
        
        Args:
            device_name: Name of the device (e.g., 'p24')
            debug: Enable debug logging
        """
        self.device_name = device_name
        self.debug = debug
        self.device = None
        self.connected = False
    
    def connect(self, fast_mode=False):
        """
        Connect to EMS device using autodetect.

        Args:
            fast_mode: Bypass stim checks for fast stimulation (<50ms)

        Returns:
            bool: True if connected successfully, False otherwise
        """
        if not EMS_AVAILABLE:
            print("Error: EMS library not available")
            return False

        try:
            print(f"Attempting to connect to EMS device ({self.device_name})...")

            # Use autodetect to find and connect to device
            self.device = EMS.autodetect(debug=self.debug, fast_mode=fast_mode)

            if self.device:
                self.connected = True
                print(f"✓ Successfully connected to EMS device")
                return True
            else:
                print("✗ Failed to detect EMS device")
                return False

        except Exception as e:
            print(f"✗ Error connecting to device: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from EMS device"""
        if self.device:
            # Add any cleanup needed
            self.device = None
            self.connected = False
            print("Disconnected from EMS device")
    
    def is_connected(self):
        """Check if device is connected"""
        return self.connected
    
    def stimulate(self, channel, intensity, pulse_width):
        """
        Perform single stimulation.

        Args:
            channel: Channel number (0-7)
            intensity: Intensity in mA
            pulse_width: Pulse width in microseconds

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            print("Error: Device not connected")
            return False

        try:
            self.device.stimulate(
                channel=channel,
                intensity=intensity,
                pulse_width=pulse_width
            )
            print(f"✓ Stimulation sent: Ch{channel}, {intensity}mA, {pulse_width}μs")
            return True

        except Exception as e:
            print(f"✗ Stimulation error: {e}")
            return False
    
    def continuous_stim(self, channel, intensity, pulse_width, pulse_count, delay):
        """
        Perform continuous stimulation.

        Args:
            channel: Channel number (0-7)
            intensity: Intensity in mA
            pulse_width: Pulse width in microseconds
            pulse_count: Number of pulses
            delay: Delay between pulses in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            print("Error: Device not connected")
            return False

        try:
            print(f"Starting continuous stimulation...")
            print(f"  Channel: {channel}")
            print(f"  Intensity: {intensity} mA")
            print(f"  Pulse Width: {pulse_width} μs")
            print(f"  Pulse Count: {pulse_count}")
            print(f"  Delay: {delay} s")

            self.device.continuous_stim(
                channel=channel,
                intensity=intensity,
                pulse_width=pulse_width,
                pulse_count=pulse_count,
                delay=delay
            )

            print(f"✓ Continuous stimulation complete")
            return True

        except Exception as e:
            print(f"✗ Continuous stimulation error: {e}")
            return False
    
    def get_device_info(self):
        """
        Get information about connected device.
        
        Returns:
            dict: Device information
        """
        return {
            "device_name": self.device_name,
            "connected": self.connected,
            "debug_mode": self.debug
        }