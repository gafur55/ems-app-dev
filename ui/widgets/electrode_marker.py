"""
Custom electrode marker widget for body diagram
"""

from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsItemGroup
from PyQt6.QtGui import QBrush, QPen, QColor, QFont
from PyQt6.QtCore import Qt
from config import settings
import math 

class ElectrodeMarker(QGraphicsItemGroup):
    """
    Visual representation of an electrode on the body diagram.
    Consists of a colored circle with a channel number inside.
    """
    
    def __init__(self, channel, x, y):
        """
        Create an electrode marker.
        
        Args:
            channel: Channel number (0-7)
            x: X position on scene
            y: Y position on scene
        """
        super().__init__()
        
        self.channel = channel
        self.x_pos = x
        self.y_pos = y
        self.active = True
        
        self.real_x_cm = None
        self.real_y_cm = None

        # Make the marker clickable
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItemGroup.GraphicsItemFlag.ItemIsSelectable)
        
        # Create the visual components
        self.create_marker()
        
        # Position the marker
        self.setPos(x, y)
    
    def create_marker(self):
        """Create the visual components of the electrode marker"""
        
        radius = settings.ELECTRODE_RADIUS
        
        # Create the circle
        # The circle is centered at (0, 0) relative to the group
        # We offset by -radius so the center is at the click point
        self.circle = QGraphicsEllipseItem(
            -radius, -radius,  # Top-left corner
            radius * 2, radius * 2  # Width and height
        )
        
        # Style the circle
        color = QColor(*settings.ELECTRODE_COLOR)  # Red
        self.circle.setBrush(QBrush(color))
        self.circle.setPen(QPen(Qt.GlobalColor.black, settings.ELECTRODE_BORDER_WIDTH))
        
        # Add circle to the group
        self.addToGroup(self.circle)
        
        # Create the channel number text
        self.text = QGraphicsTextItem(str(self.channel))
        font = QFont("Arial", 12, QFont.Weight.Bold)
        self.text.setFont(font)
        self.text.setDefaultTextColor(Qt.GlobalColor.white)
        
        # Center the text within the circle
        text_rect = self.text.boundingRect()
        text_x = -text_rect.width() / 2
        text_y = -text_rect.height() / 2
        self.text.setPos(text_x, text_y)
        
        # Add text to the group
        self.addToGroup(self.text)
    
    def set_active(self, active):
        """
        Set electrode active/inactive state.
        Changes color based on state.
        
        Args:
            active: True for active (green), False for inactive (gray)
        """
        self.active = active
        
        if active:
            color = QColor(*settings.ELECTRODE_ACTIVE_COLOR)
        else:
            color = QColor(*settings.ELECTRODE_INACTIVE_COLOR)
        
        self.circle.setBrush(QBrush(color))
    
    def get_data(self):
        """
        Get electrode data for saving/export.
        
        Returns:
            dict: Electrode data (channel, position, active state)
        """
        return {
            "channel": self.channel,
            "x": self.x_pos,
            "y": self.y_pos,
            "active": self.active
        }
    
    def set_real_coordinates(self, real_x_cm, real_y_cm):
        """
        Set and display real-world coordinates.
        
        Args:
            real_x_cm: X position in cm from wrist
            real_y_cm: Y position in cm from centerline
        """
        self.real_x_cm = real_x_cm
        self.real_y_cm = real_y_cm
        
        # Calculate distance from wrist
        distance = math.sqrt(real_x_cm**2 + real_y_cm**2)
        
        # Update tooltip to show both coordinate systems
        tooltip_text = (
            f"Channel {self.channel}\n"
            f"Pixel: ({self.x_pos:.0f}, {self.y_pos:.0f})\n"
            f"Real: ({real_x_cm:.1f}cm, {real_y_cm:.1f}cm)\n"
            f"Distance from wrist: {distance:.1f}cm"
        )
        self.setToolTip(tooltip_text)