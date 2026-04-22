"""
Custom QGraphicsView for body diagram display and interaction.

Calibration is now fully automatic via ArUco — no manual clicks.
The live forearm feed (annotated with ArUco overlays) IS the calibration UI.

Flow:
    1. ForearmCamera runs ArUco detection continuously
    2. "Place Electrodes" → reads ArUco state → calibrates instantly
    3. Electrode ArUco markers (IDs 4, 5) are auto-placed if detected
    4. User can still click manually to add/remove electrodes after auto-placement
"""

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtGui     import QPainter
from PyQt6.QtCore    import Qt

from ui.managers import SceneManager, ElectrodeManager, CalibrationManager


class BodyDiagramView(QGraphicsView):
    """
    Graphics view for body diagram interactions.
    Delegates to SceneManager, ElectrodeManager, CalibrationManager.
    """

    def __init__(self):
        super().__init__()

        self._scene = QGraphicsScene()
        self.setScene(self._scene)

        self.scene_manager       = SceneManager(self._scene)
        self.electrode_manager   = ElectrodeManager(self._scene)
        self.calibration_manager = CalibrationManager(self._scene)

        self.calibration_manager.calibration_complete.connect(
            self._on_calibration_complete
        )

        # Electrode placement enabled only after calibration
        self._placement_enabled = False

        self._setup_view()
        self.scene_manager.load_body_diagram()

    def _setup_view(self) -> None:
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(Qt.GlobalColor.white)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def _on_calibration_complete(self, mapper):
        self.electrode_manager.set_coordinate_mapper(mapper)
        self._placement_enabled = True
        print("✓ Calibration complete — electrode placement enabled")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_session(self, session) -> None:
        self.electrode_manager.set_session(session)
        self.calibration_manager.set_session(session)
        print(f"Graphics view connected to session: {session.participant_id}")

    def update_live_frame(self, cv_image) -> None:
        """Push latest ArUco-annotated camera frame to background."""
        self.scene_manager.update_live_frame(cv_image)

    def load_captured_image(self, image) -> None:
        self.scene_manager.load_captured_image(image)

    def calibrate_from_aruco(self, aruco_state: dict) -> bool:
        """
        Run ArUco calibration from the given state dict.
        Draws reference markers on scene and enables electrode placement.

        Returns True on success.
        """
        self._placement_enabled = False
        return self.calibration_manager.complete_from_aruco(aruco_state)

    def auto_place_electrodes_from_aruco(self, aruco_state: dict) -> int:
        """
        Place electrode markers automatically from ArUco electrode detections
        (marker IDs 4, 5, …). Clears existing electrodes first.

        Returns number of electrodes placed.
        """
        if not self._placement_enabled:
            print("✗ Cannot auto-place — calibration not complete.")
            return 0

        electrodes = aruco_state.get("electrodes", {})
        if not electrodes:
            print("  No electrode markers detected (IDs 4, 5).")
            return 0

        self.electrode_manager.clear_all()
        placed = 0
        for eid, info in electrodes.items():
            px, py = float(info["pixel"][0]), float(info["pixel"][1])
            if self.electrode_manager.place_electrode(px, py):
                placed += 1
                print(f"  E{eid} auto-placed: "
                      f"{info['down_mm']:.1f}mm from elbow | "
                      f"{abs(info['lateral_mm']):.1f}mm toward {info['side']}")

        print(f"✓ {placed} electrode(s) auto-placed from ArUco")
        return placed

    def clear_electrodes(self) -> None:
        self.electrode_manager.clear_all()
        self._placement_enabled = False

    def clear_calibration(self) -> None:
        self.calibration_manager.reset()
        self._placement_enabled = False

    def reset_all(self) -> None:
        self.electrode_manager.clear_all()
        self.calibration_manager.reset()
        self._placement_enabled = False
        self.scene_manager.clear()
        self.scene_manager.load_body_diagram()

    def toggle_grid(self, visible: bool) -> None:
        self.scene_manager.toggle_grid(visible)

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        view_pos  = event.pos()
        scene_pos = self.mapToScene(view_pos)

        if event.button() == Qt.MouseButton.RightButton:
            if self._placement_enabled:
                self._handle_right_click(scene_pos)
            return

        x, y = scene_pos.x(), scene_pos.y()

        if self._placement_enabled:
            if self.electrode_manager.can_place_electrode:
                self.electrode_manager.place_electrode(x, y)
            else:
                print("Maximum electrodes reached.")
        else:
            print("[blocked] Click 'Place Electrodes' first — ArUco must detect all 4 markers.")

        super().mousePressEvent(event)

    def _handle_right_click(self, scene_pos) -> None:
        marker = self.electrode_manager.find_electrode_at(scene_pos, self.transform())
        if marker:
            self.electrode_manager.delete_electrode(marker)

    # ── Compatibility ─────────────────────────────────────────────────────────

    @property
    def coordinate_mapper(self):
        return self.electrode_manager.coordinate_mapper

    @property
    def electrodes(self):
        return self.electrode_manager.electrodes