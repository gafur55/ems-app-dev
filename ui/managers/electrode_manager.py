"""
Manages electrode placement, deletion, and channel tracking
"""

from PyQt6.QtWidgets import QGraphicsScene
from typing import Optional, List
from config import settings
from ui.widgets.electrode_marker import ElectrodeMarker
from data.models import Session, Electrode


class ElectrodeManager:
    """Handles electrode placement and management"""
    
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.electrodes: List[ElectrodeMarker] = []
        self.current_channel = 0
        self.channel_electrode_count = 0
        self.session: Optional[Session] = None
        self.coordinate_mapper = None
    
    @property
    def electrode_count(self) -> int:
        """Total number of placed electrodes"""
        return len(self.electrodes)
    
    @property
    def can_place_electrode(self) -> bool:
        """Check if more electrodes can be placed"""
        return self.electrode_count < settings.MAX_ELECTRODES
    

    def set_session(self, session: Session) -> None:
        """
        Connect to a session for data storage.
        
        Args:
            session: Session object to store electrode data
        """
        self.session = session
        print(f"Electrode manager connected to session: {session.participant_id}")
    

    def set_coordinate_mapper(self, mapper) -> None:
        """
        Set the coordinate mapper for real-world coordinates.
        
        Args:
            mapper: AnatomicalCoordinateMapper instance
        """
        self.coordinate_mapper = mapper
        print("Electrode manager received coordinate mapper")
    

    def place_electrode(self, x: float, y: float) -> bool:
        """
        Place an electrode at the given position.
        
        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            
        Returns:
            True if electrode was placed, False if max reached
        """
        if not self.can_place_electrode:
            print(f"Maximum electrodes reached ({settings.MAX_ELECTRODES})")
            return False
        
        # Create marker
        marker = ElectrodeMarker(self.current_channel, x, y)
        
        # Add coordinate data if mapper available
        if self.coordinate_mapper:
            self._add_coordinate_data(marker, x, y)
        else:
            print(f"  Warning: No coordinate system. Pixel position: ({x:.1f}, {y:.1f})")
        
        # Add to scene and track
        self.scene.addItem(marker)
        self.electrodes.append(marker)
        self.channel_electrode_count += 1
        
        print(f"Placed electrode Ch{self.current_channel} "
              f"({self.channel_electrode_count}/{settings.ELECTRODES_PER_CHANNEL}) "
              f"at ({x:.1f}, {y:.1f})")
        
        # Check if channel pair complete
        if self.channel_electrode_count >= settings.ELECTRODES_PER_CHANNEL:
            print(f"✓ Channel {self.current_channel} pair complete!")
            self.current_channel += 1
            self.channel_electrode_count = 0
        
        return True
    
    
    def _add_coordinate_data(self, marker: ElectrodeMarker, x: float, y: float) -> None:
        """
        Add real-world coordinate data to electrode.
        
        Args:
            marker: ElectrodeMarker to update
            x: X pixel coordinate
            y: Y pixel coordinate
        """
        s, t, real_x_cm, real_y_cm = self.coordinate_mapper.pixel_to_anatomical(x, y)
        marker.set_real_coordinates(real_x_cm, real_y_cm)
        
        # Create data model
        electrode_data = Electrode(self.current_channel, x, y)
        electrode_data.real_x_cm = real_x_cm
        electrode_data.real_y_cm = real_y_cm
        electrode_data.normalized_longitudinal = s
        electrode_data.normalized_lateral = t
        electrode_data.distance_from_wrist_cm = self.coordinate_mapper.distance_from_origin(
            real_x_cm, real_y_cm
        )
        
        # Store in session
        if self.session:
            self.session.electrodes.append(electrode_data)
        
        coord_str = self.coordinate_mapper.format_coordinates(s, t, real_x_cm, real_y_cm)
        print(f"✓ Electrode Ch{self.current_channel} placed at:\n{coord_str}")
    

    def delete_electrode(self, marker: ElectrodeMarker) -> None:
        """
        Remove an electrode marker.
        
        Args:
            marker: ElectrodeMarker to delete
        """
        channel = marker.channel
        
        self.scene.removeItem(marker)
        if marker in self.electrodes:
            self.electrodes.remove(marker)
        
        self._recalculate_channel_state()
        print(f"Deleted electrode from Channel {channel}")
    

    def _recalculate_channel_state(self) -> None:
        """Recalculate channel state after deletion"""
        total = len(self.electrodes)
        self.current_channel = total // settings.ELECTRODES_PER_CHANNEL
        self.channel_electrode_count = total % settings.ELECTRODES_PER_CHANNEL
        
        print(f"State: {total} electrodes, Channel {self.current_channel}, "
              f"{self.channel_electrode_count}/{settings.ELECTRODES_PER_CHANNEL}")
    

    def clear_all(self) -> None:
        """Remove all electrodes and reset state"""
        for marker in self.electrodes:
            self.scene.removeItem(marker)
        
        self.electrodes.clear()
        self.current_channel = 0
        self.channel_electrode_count = 0
        print("All electrodes cleared")
    

    def find_electrode_at(self, scene_pos, transform) -> Optional[ElectrodeMarker]:
        """
        Find electrode marker at a scene position.
        
        Args:
            scene_pos: Position in scene coordinates
            transform: View transform for hit testing
            
        Returns:
            ElectrodeMarker if found, None otherwise
        """
        item = self.scene.itemAt(scene_pos, transform)
        
        if item:
            # Check parent since click might be on circle or text child
            parent = item.parentItem()
            if isinstance(parent, ElectrodeMarker):
                return parent
            if isinstance(item, ElectrodeMarker):
                return item
        
        return None