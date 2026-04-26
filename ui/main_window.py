"""
Main application window for EMS Control GUI

Flow:
1. Sign-in dialog → select/create participant (personal info stored in DB)
2. Main GUI → device selection, live forearm camera, calibrate, place electrodes, stimulate
3. All data auto-saved to SQLite database

Tracking setup (3 cameras):
    Camera 0 (iPhone #1)  → HandTracker    → finger joint angles (MCP, PIP, DIP)
    Camera 1 (laptop)     → PoseTracker    → wrist angle (elbow→wrist→mid)
    Camera 2 (iPhone #2)  → ForearmCamera  → live forearm view for electrode placement
"""

from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QDialog
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import threading
import time
import os
from ui.graphics_view import BodyDiagramView
from ui.widgets.parameter_panel import ParameterPanel
from ui.widgets.electrode_placement_panel import ElectrodePlacementPanel
from data.models import Session
from ui.widgets.stimulation_panel import StimulationPanel
from ui.widgets.sign_in_dialog import SignInDialog
from config import settings
from core.ems_controller import EMSController

# Split trackers + preview
from core.vision.pose_tracker import PoseTracker
from core.vision.hand_tracker import HandTracker
from core.vision.forearm_camera import ForearmCamera
from ui.widgets.tracking_preview import TrackingPreviewWindow

# Audio recording
from core.session_recorder import SessionRecorder
from core.session_transcriber import SessionTranscriber

# Database
from data.database import EMSDatabase
import numpy as np

# =====================================================================
# CAMERA CONFIGURATION — change these to match your setup
# =====================================================================
HAND_CAMERA    = 1
POSE_CAMERA    = 2
FOREARM_CAMERA = 0

ARM_SIDE = "left"  # mirror effect (left here is right in reality)


class EMSWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.current_session = None
        self.ems_controller = None

        # Split trackers
        self.pose_tracker = None
        self.hand_tracker = None
        self.pose_preview = None
        self.hand_preview = None

        # Forearm camera (live view for electrode placement)
        self.forearm_camera = None
        self.forearm_timer  = None
        self.forearm_paused = False

        # Audio recorder
        self.session_recorder    = SessionRecorder(save_dir="data/recordings")
        self.session_transcriber = None
        self.recording_id        = None

        import atexit
        atexit.register(self._emergency_stop_recording)

        # Database
        self.db            = EMSDatabase()
        self.db_session_id = None

        # Participant data (loaded from sign-in)
        self.participant       = None
        self.forearm_length_cm = None

        self.setup_ui()
        self._show_sign_in()

    # =====================================================================
    # UI Setup
    # =====================================================================
    def _setup_hotkeys(self) -> None:
        """Global hotkeys for data collection."""
        from PyQt6.QtGui import QShortcut, QKeySequence
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QLineEdit, QApplication

        def make_shortcut(key, callback, description=""):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)

            def guarded():
                # Don't fire while a text input is focused — let the user type
                focused = QApplication.focusWidget()
                if isinstance(focused, QLineEdit):
                    return
                callback()

            sc.activated.connect(guarded)
            return sc

        self._sc_stim   = make_shortcut("Space", self.stimulation_panel.trigger_stimulate)
        self._sc_intens = make_shortcut("I",     lambda: self.stimulation_panel.increment_intensity(1))
        self._sc_pw     = make_shortcut("W",     lambda: self.stimulation_panel.increment_pulse_width(10))
        self._sc_count  = make_shortcut("C",     lambda: self.stimulation_panel.increment_pulse_count(5))
        self._sc_reset  = make_shortcut("R",     self.stimulation_panel.reset_to_defaults)
        self._sc_intens_dn = make_shortcut("U", lambda: self.stimulation_panel.increment_intensity(-1))
        self._sc_pw_dn     = make_shortcut("Q", lambda: self.stimulation_panel.increment_pulse_width(-10))
        self._sc_count_dn  = make_shortcut("X", lambda: self.stimulation_panel.increment_pulse_count(-5))
        self._sc_place = make_shortcut("P", self._on_place_electrodes)
        print("✓ Hotkeys: Space=stim | P=place | I/U=±1mA | W/Q=±10μs | C/X=±5pulses | R=reset")



    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle(settings.APP_NAME)
        self.setGeometry(100, 100, 1400, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        central_widget.setLayout(outer_layout)

        # --- Participant header bar ---
        self.header_bar = QWidget()
        self.header_bar.setStyleSheet("background-color: #1e3a5f; padding: 6px 12px;")
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 6, 12, 6)
        self.header_bar.setLayout(header_layout)

        self.participant_label = QLabel("No participant signed in")
        self.participant_label.setStyleSheet("color: white; font-size: 13px;")
        header_layout.addWidget(self.participant_label)

        header_layout.addStretch()

        from PyQt6.QtWidgets import QPushButton
        self.switch_user_btn = QPushButton("Switch Participant")
        self.switch_user_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #93c5fd;
                border: 1px solid #93c5fd;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #2563eb; color: white; }
        """)
        self.switch_user_btn.clicked.connect(self._show_sign_in)
        header_layout.addWidget(self.switch_user_btn)

        outer_layout.addWidget(self.header_bar)

        # --- Main content ---
        main_content = QWidget()
        main_layout = QHBoxLayout()
        main_content.setLayout(main_layout)
        outer_layout.addWidget(main_content, stretch=1)

        self._setup_body_view(main_layout)
        self._setup_right_panel(main_layout)

        self.body_view.setEnabled(False)

    def _setup_body_view(self, parent_layout):
        """Create body diagram view with PyQt ArUco status label below it."""
        container = QWidget()
        container.setMinimumWidth(500)
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container.setLayout(container_layout)

        # Body diagram view
        self.body_view = BodyDiagramView()
        container_layout.addWidget(self.body_view, stretch=1)

        # ArUco status label — PyQt rendered, always readable
        self.aruco_status_label = QLabel("Camera not active")
        self.aruco_status_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 200);
                color: #00ff88;
                font-family: Helvetica, Arial, sans-serif;
                font-size: 12px;
                padding: 5px 10px;
            }
        """)
        self.aruco_status_label.setWordWrap(True)
        self.aruco_status_label.setFixedHeight(28)
        container_layout.addWidget(self.aruco_status_label)

        parent_layout.addWidget(container, stretch=1)

        # Calibration signal
        self.body_view.calibration_manager.calibration_complete.connect(
            self._on_calibration_complete
        )

    def _setup_right_panel(self, parent_layout):
        """Create the parameter and stimulation panels"""
        right_layout = QVBoxLayout()

        self._setup_parameter_panel(right_layout)
        self._setup_electrode_button(right_layout)
        self._setup_stimulation_panel(right_layout)

        right_layout.addStretch()

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(600)
        parent_layout.addWidget(right_widget, stretch=0)

    def _setup_parameter_panel(self, parent_layout):
        """Create and configure the parameter panel"""
        self.parameter_panel = ParameterPanel()
        self.parameter_panel.session_started.connect(self.on_session_started)
        self.parameter_panel.session_stopped.connect(self.on_session_stopped)

        cc = self.parameter_panel.capture_controls
        cc.place_electrodes_requested.connect(self._on_place_electrodes)

        parent_layout.addWidget(self.parameter_panel)

    def _setup_stimulation_panel(self, parent_layout):
        """Create and configure the stimulation panel"""
        self.stimulation_panel = StimulationPanel()
        self._setup_hotkeys()
        self.stimulation_panel.stimulate_requested.connect(self.on_stimulate_requested)
        parent_layout.addWidget(self.stimulation_panel)
        self._set_stimulate_enabled(False)

    def _setup_electrode_button(self, parent_layout):
        """Create the electrode placement panel."""
        self.electrode_panel = ElectrodePlacementPanel()
        self.electrode_panel.electrodes_confirmed.connect(self.on_electrodes_confirmed)
        parent_layout.addWidget(self.electrode_panel)

    # =====================================================================
    # Sign-in
    # =====================================================================

    def _show_sign_in(self):
        """Show the participant sign-in dialog."""
        if self.current_session:
            self.on_session_stopped()

        dialog = SignInDialog(self.db, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            self.participant = dialog.get_participant_data()
            pid  = self.participant["participant_id"]
            name = self.participant.get("name") or pid

            parts = [f"{name} ({pid})"]
            if self.participant.get("age"):
                parts.append(f"Age: {self.participant['age']}")
            if self.participant.get("experience_level"):
                parts.append(f"Exp: {self.participant['experience_level']}")
            self.participant_label.setText("  |  ".join(parts))

            print(f"\n✓ Signed in as {pid}")

            # self.parameter_panel.capture_controls.set_arm_length(
            #     self.participant.get("arm_length")
            # )
        else:
            if self.participant is None:
                self.participant_label.setText(
                    "No participant — click 'Switch Participant'"
                )

    # =====================================================================
    # Session lifecycle
    # =====================================================================

    def on_session_started(self, params):
        """Handle session start event."""
        if not self.participant:
            print("✗ No participant signed in — showing sign-in dialog")
            self._show_sign_in()
            if not self.participant:
                return

        pid         = self.participant["participant_id"]
        device_name = params["device_name"]

        self.current_session = Session(
            participant_id=pid,
            device_name=device_name
        )
        self.current_session.age                        = self.participant.get("age")
        self.current_session.arm_width                  = self.participant.get("arm_width")
        self.current_session.arm_length                 = self.participant.get("arm_length")
        self.current_session.skin_resistance_100hz_kohm = self.participant.get("skin_resistance_100hz_kohm")
        self.current_session.skin_resistance_1khz_kohm  = self.participant.get("skin_resistance_1khz_kohm")
        self.current_session.skin_resistance_10khz_kohm = self.participant.get("skin_resistance_10khz_kohm")
        self.current_session.skin_resistance_100khz_kohm= self.participant.get("skin_resistance_100khz_kohm")
        self.current_session.pain_threshold_ma          = self.participant.get("pain_threshold_ma")
        self.current_session.experience_level           = self.participant.get("experience_level")

        print(f"\n=== Session Created ===")
        print(f"Participant: {pid}")
        print(f"Device: {device_name}")

        self.db_session_id = self.db.create_session(
            participant_id=pid,
            device_name=device_name,
            forearm_length_cm=self.participant.get("arm_length"),
        )

        print(f"\n=== Connecting to EMS Device ===")

        if self.session_transcriber is None:
            self.session_transcriber = SessionTranscriber(db=self.db)

        self.ems_controller = EMSController(
            device_name=params["device_name"],
            debug=True
        )

        if self.ems_controller.connect():
            print("✓ Device connected - Ready to stimulate!")

            self.body_view.setEnabled(True)
            self.stimulation_panel.setEnabled(True)
            self.body_view.set_session(self.current_session)

            self._start_trackers()
            self._start_recording()

            self.electrode_panel.setEnabled(True)
            self._set_stimulate_enabled(False)

            self._set_phase_status("Click 'Place Electrodes' to begin", "#FF9800")
            print("Ready to place electrodes!")
        else:
            print("✗ Failed to connect to device")
            print("  Check that device is plugged in and try again")

    def on_session_stopped(self):
        """Handle session stop."""
        print("\n=== Stopping Session ===")

        self._stop_recording()
        self._stop_trackers()

        if self.ems_controller and self.ems_controller.is_connected():
            self.ems_controller.disconnect()
            print("✓ Device disconnected")

        self.body_view.clear_electrodes()
        print("✓ Electrodes cleared")

        if self.db_session_id:
            max_intensity = self.db.get_session_max_intensity(self.db_session_id)
            if max_intensity is not None and self.participant:
                pid = self.participant["participant_id"]
                self.db.update_pain_threshold_ma(pid, max_intensity)
                print(f"  Pain threshold updated: {max_intensity} mA")

            self.db.end_session(self.db_session_id)
            self.db.print_stats()
            self.db_session_id = None

        self.body_view.setEnabled(False)
        self.stimulation_panel.setEnabled(False)
        self._set_stimulate_enabled(False)
        self.electrode_panel.reset()
        self.electrode_panel.setEnabled(False)

        self.current_session = None
        self.ems_controller  = None
        self.aruco_status_label.setText("Camera not active")

        print("✓ Session stopped - ready to start new session")

    # =====================================================================
    # Status labels
    # =====================================================================

    def _set_phase_status(self, message: str, color: str = "gray") -> None:
        """Update the single status label shown on the right panel."""
        self.parameter_panel.status_label.setText(message)
        self.parameter_panel.status_label.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 13px;"
        )

    def _update_aruco_label(self, state: dict) -> None:
        """Update the PyQt ArUco status label below the body view."""
        if not state.get("mat_ready", False):
            fills  = state.get("fills", {})
            labels = {4: "TL", 0: "TR", 2: "BR", 3: "BL"}
            parts  = []
            for mid in [4, 0, 2, 3]:
                count = fills.get(mid, 0)
                parts.append(f"ID{mid}({labels.get(mid,'?')}): {count}/15")
            self.aruco_status_label.setText(
                "Waiting for mat corners — " + "  |  ".join(parts)
            )
            self.aruco_status_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0,0,0,200);
                    color: #ff9800;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    padding: 5px 10px;
                }
            """)
        elif state.get("ready"):
            px_per_mm  = state.get("px_per_mm", 0)
            electrodes = state.get("electrodes", {})
            e_text     = f"  |  Electrodes: {len(electrodes)}" if electrodes else ""
            self.aruco_status_label.setText(
                f"✓ Calibrated  |  "
                f"{settings.MAT_WIDTH_MM:.0f}×{settings.MAT_HEIGHT_MM:.0f}mm  |  "
                f"{px_per_mm:.2f} px/mm{e_text}"
            )
            self.aruco_status_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0,0,0,200);
                    color: #00ff88;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    padding: 5px 10px;
                }
            """)
        else:
            self.aruco_status_label.setText(
                "✓ Mat ready — waiting for wrist marker (ID 1)"
            )
            self.aruco_status_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0,0,0,200);
                    color: #60a5fa;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    padding: 5px 10px;
                }
            """)

    # =====================================================================
    # Electrode placement gate
    # =====================================================================

    def _set_stimulate_enabled(self, enabled: bool) -> None:
        """Enable/disable the STIMULATE button."""
        self.stimulation_panel.stimulate_button.setEnabled(enabled)

    def on_electrodes_confirmed(self):
        """Handle 'Confirm Electrode Placement' button click."""
        electrode_count = len(self.body_view.electrodes)

        if electrode_count < 2:
            print(f"✗ Need at least 2 electrodes, currently: {electrode_count}")
            self.electrode_panel.set_error(
                f"Place at least 2 electrodes first ({electrode_count} placed)"
            )
            self._set_phase_status(
                f"Place at least 2 electrodes ({electrode_count} placed)", "#dc2626"
            )
            return

        print(f"\n=== Electrodes Confirmed ({electrode_count} placed) ===")

        self._set_stimulate_enabled(True)
        self.electrode_panel.set_confirmed(electrode_count)
        self._set_phase_status("Ready to stimulate!", "#16a34a")

        self._resume_forearm_feed()
        print("✓ Stimulation enabled — ready to go!")

    # =====================================================================
    # Stimulation with dual-camera movement tracking
    # =====================================================================

    def on_stimulate_requested(self, params):
        """
        Handle stimulation request.
        Flow:
            1. Capture baselines (pose + hand)
            2. Record stimulation to DB (+ electrode positions)
            3. Start recording + fire stimulation simultaneously
            4. Save combined movement result to DB
        """
        channel     = params["channel"]
        intensity   = params["intensity"]
        pulse_width = params["pulse_width"]
        pulse_count = params["pulse_count"]
        delay       = params["delay"]

        print(f"\n=== Stimulation Requested ===")
        print(f"Channel: {channel}")
        print(f"Intensity: {intensity} mA")
        print(f"Pulse Width: {pulse_width} μs")
        print(f"Pulse Count: {pulse_count}")
        print(f"Delay: {delay} ms")

        if not self.ems_controller or not self.ems_controller.is_connected():
            self._set_phase_status("Error: Device not connected", "#dc2626")
            print("✗ Cannot stimulate - device not connected")
            return

        # ---- STEP 1: Capture baselines ----
        pose_baseline_ok = False
        hand_baseline_ok = False

        if self.pose_tracker and self.pose_tracker.is_running:
            print("\n--- Capturing pose baseline (wrist) ---")
            pose_baseline_ok = self.pose_tracker.capture_baseline()
            if not pose_baseline_ok:
                print("⚠ No pose detected — wrist tracking unavailable")

        if self.hand_tracker and self.hand_tracker.is_running:
            print("--- Capturing hand baseline (fingers) ---")
            hand_baseline_ok = self.hand_tracker.capture_baseline()
            if not hand_baseline_ok:
                print("⚠ No hand detected — finger tracking unavailable")

        if not pose_baseline_ok and not hand_baseline_ok:
            print("⚠ No tracking baselines — stimulating without tracking")

        
        # ---- STEP 2: Record stimulation to database ----
        stim_id = None
        if self.db_session_id:
            stim_id = self.db.record_stimulation(
                session_id=self.db_session_id,
                channel=channel,
                intensity_ma=intensity,
                pulse_width_us=pulse_width,
                pulse_count=pulse_count,
                delay_ms=delay,
            )

        self._capture_forearm_image(stim_id)


        # Read live ArUco positions at moment of stimulation
        if self.forearm_camera and self._electrode_placement and stim_id:
            state      = self.forearm_camera.get_aruco_state()
            electrodes = state.get("electrodes", {})
            H          = state.get("homography")
            wrist_px   = state.get("wrist_center_px")

            if H is None or wrist_px is None:
                print("⚠ Mat/wrist not ready — skipping electrode DB save")
            else:
                def px_to_mat_mm(px_pt):
                    """Convert camera pixel to mat mm via homography."""
                    pt     = np.array([[[float(px_pt[0]), float(px_pt[1])]]], dtype=np.float32)
                    result = cv2.perspectiveTransform(pt, H)
                    return np.array([result[0][0][0], result[0][0][1]], dtype=np.float32)

                import cv2
                wrist_mm = px_to_mat_mm(wrist_px)
                print(f"  Wrist in mat frame: ({wrist_mm[0]:.1f}, {wrist_mm[1]:.1f}) mm")

                for i, (eid, info) in enumerate(electrodes.items()):
                    e_px    = info["pixel"]
                    e_mm    = px_to_mat_mm(e_px)
                    diff_mm = e_mm - wrist_mm

                    # Straight-line distance from wrist to electrode in mat mm
                    distance_mm = float(np.linalg.norm(diff_mm))

                    # Mat-frame offset components
                    dx_mm = float(diff_mm[0])   # positive = right in mat frame
                    dy_mm = float(diff_mm[1])   # positive = down in mat frame

                    print(f"  DB: E{eid} — "
                        f"dist={distance_mm:.1f}mm | "
                        f"dx={dx_mm:.1f}mm, dy={dy_mm:.1f}mm | "
                        f"mat=({e_mm[0]:.1f}, {e_mm[1]:.1f})mm")

                    self.db.record_electrode(
                        session_id=self.db_session_id,
                        channel=i,
                        pixel_x=float(e_px[0]),
                        pixel_y=float(e_px[1]),
                        stim_id=stim_id,
                        real_x_cm=dx_mm / 10.0,   # mat x offset from wrist
                        real_y_cm=dy_mm / 10.0,   # mat y offset from wrist
                        placement_method="aruco_mat",
                    )


        # ---- STEP 3: Start recording + stimulate simultaneously ----
        stim_duration_s    = (pulse_count * delay) / 1000.0
        recording_duration = stim_duration_s + 1.5

        wrist_result_box  = [None]
        finger_result_box = [None]

        pose_thread = None
        if pose_baseline_ok and self.pose_tracker:
            print(f"\n--- Recording wrist movement for {recording_duration:.1f}s ---")

            def _record_pose():
                wrist_result_box[0] = self.pose_tracker.measure_movement(
                    recording_duration=recording_duration,
                    channel=channel,
                    intensity=intensity,
                    pulse_width=pulse_width,
                )

            pose_thread = threading.Thread(target=_record_pose, daemon=True)
            pose_thread.start()

        hand_thread = None
        if hand_baseline_ok and self.hand_tracker:
            print(f"--- Recording finger movement for {recording_duration:.1f}s ---")

            def _record_hand():
                finger_result_box[0] = self.hand_tracker.measure_movement(
                    recording_duration=recording_duration,
                    channel=channel,
                    intensity=intensity,
                    pulse_width=pulse_width,
                )

            hand_thread = threading.Thread(target=_record_hand, daemon=True)
            hand_thread.start()

        time.sleep(0.05)

        success = self.ems_controller.continuous_stim(
            channel=channel,
            intensity=intensity,
            pulse_width=pulse_width,
            pulse_count=pulse_count,
            delay=delay / 1000.0
        )

        if not success:
            self._set_phase_status("Stimulation failed ✗", "#dc2626")
            return

        # ---- STEP 4: Wait for recordings + save to database ----
        wrist_result  = None
        finger_result = None

        if pose_thread:
            pose_thread.join(timeout=5.0)
            wrist_result = wrist_result_box[0]

        if hand_thread:
            hand_thread.join(timeout=5.0)
            finger_result = finger_result_box[0]

        status_parts = ["Stimulation complete ✓"]

        if wrist_result and wrist_result.pose_detected:
            status_parts.append(f"Wrist: {wrist_result.wrist_feedback}")

        if finger_result and finger_result.hand_detected:
            status_parts.append(
                f"Fingers: {finger_result.primary_finger} "
                f"{finger_result.primary_finger_movement:.1f}° | "
                f"Peak at {finger_result.latency_ms:.0f}ms"
            )

        if not wrist_result and not finger_result:
            status_parts.append("(movement not captured)")

        self._set_phase_status(" | ".join(status_parts), "#16a34a")

        if stim_id:
            combined = _CombinedMovementResult(wrist_result, finger_result)
            print(f"\n  DB DEBUG: wrist_delta={combined.wrist_angle_delta}, "
                  f"wrist_dir={combined.wrist_direction}, "
                  f"finger_total={combined.total_finger_movement}, "
                  f"hand_detected={combined.hand_detected}")
            self.db.record_movement(stim_id, combined)

    # =====================================================================
    # Tracker management
    # =====================================================================

    def _start_trackers(self) -> None:
        """Start both movement trackers and open preview windows."""
        arm_side = ARM_SIDE

        print("\n=== Starting Movement Trackers ===")

        # --- Forearm camera: live forearm view ---
        if FOREARM_CAMERA >= 0:
            self.forearm_camera = ForearmCamera(camera_index=FOREARM_CAMERA)
            if self.forearm_camera.start():
                if self.participant and self.participant.get("arm_length"):
                    self.forearm_camera.set_arm_length(
                        self.participant["arm_length"]
                    )
                print("✓ Forearm camera ready (live view)")
                self.forearm_timer = QTimer()
                self.forearm_timer.timeout.connect(self._update_forearm_feed)
                self.forearm_timer.start(33)
                self.forearm_paused = False
                self.parameter_panel.capture_controls.set_camera_active(True)
            else:
                print("⚠ Forearm camera failed to start")
                self.forearm_camera = None
        else:
            print("  Forearm camera disabled (FOREARM_CAMERA = -1)")
            self.forearm_camera = None


        self.electrode_monitor = ElectrodePositionMonitor(self)
        self.electrode_monitor.show()


        # --- Pose tracker: wrist angle ---
        pose_save_dir = f"captures/pose_{self.db_session_id:03d}"
        self.pose_tracker = PoseTracker(
            camera_index=POSE_CAMERA,
            arm=arm_side,
            smoothing_window=5,
            noise_threshold=2.0,
            show_preview=True,
            save_dir=pose_save_dir,
        )

        if self.pose_tracker.start():
            print("✓ Pose tracker ready (wrist angle)")
            self.pose_preview = TrackingPreviewWindow(self.pose_tracker)
            self.pose_preview.setWindowTitle(
                "Wrist Tracking — Pose (auto arm) [Laptop Camera]"
            )
            self.pose_preview.show()
            print("✓ Pose preview window opened")
        else:
            print("⚠ Pose tracker failed to start — wrist tracking unavailable")
            self.pose_tracker = None

        # --- Hand tracker: finger angles ---
        hand_save_dir = f"captures/hand_{self.db_session_id:03d}"
        self.hand_tracker = HandTracker(
            camera_index=HAND_CAMERA,
            smoothing_window=5,
            show_preview=True,
            save_dir=hand_save_dir,
        )

        if self.hand_tracker.start():
            print("✓ Hand tracker ready (finger angles)")
            self.hand_preview = TrackingPreviewWindow(self.hand_tracker)
            self.hand_preview.setWindowTitle(
                "Finger Tracking — Hand [iPhone Camera]"
            )
            self.hand_preview.show()
            print("✓ Hand preview window opened")
        else:
            print("⚠ Hand tracker failed to start — finger tracking unavailable")
            self.hand_tracker = None

    def _stop_trackers(self) -> None:
        """Stop all trackers, forearm camera, and close preview windows."""
        if self.forearm_timer:
            self.forearm_timer.stop()
            self.forearm_timer = None

        if self.forearm_camera:
            self.forearm_camera.stop()
            self.forearm_camera = None
            print("✓ Forearm camera stopped")

        self.parameter_panel.capture_controls.set_camera_active(False)

        if self.pose_preview:
            self.pose_preview.stop()
            self.pose_preview.close()
            self.pose_preview = None
            print("✓ Pose preview closed")

        if self.hand_preview:
            self.hand_preview.stop()
            self.hand_preview.close()
            self.hand_preview = None
            print("✓ Hand preview closed")

        if self.pose_tracker:
            self.pose_tracker.stop()
            self.pose_tracker = None
            print("✓ Pose tracker stopped")

        if self.hand_tracker:
            self.hand_tracker.stop()
            self.hand_tracker = None
            print("✓ Hand tracker stopped")

        if hasattr(self, 'electrode_monitor') and self.electrode_monitor:
            self.electrode_monitor.close()
            self.electrode_monitor = None
    # =====================================================================
    # Audio recording
    # =====================================================================

    def _start_recording(self):
        """Start audio recording for the current session."""
        if not self.db_session_id or not self.participant:
            return

        pid      = self.participant["participant_id"]
        filepath = self.session_recorder.start(
            session_id=self.db_session_id,
            participant_id=pid,
        )

        if filepath:
            self.recording_id = self.db.start_recording(
                session_id=self.db_session_id,
                participant_id=pid,
                filepath=filepath,
            )

    def _stop_recording(self):
        """Stop audio recording, finalize in DB, and trigger transcription."""
        if not self.session_recorder.is_recording:
            return

        filepath       = self.session_recorder.stop()
        recording_id   = self.recording_id
        session_id     = self.db_session_id
        participant_id = self.participant["participant_id"] if self.participant else ""

        if recording_id and filepath and os.path.exists(filepath):
            duration  = self.session_recorder._frames_written / self.session_recorder.SAMPLE_RATE
            file_size = os.path.getsize(filepath)
            self.db.finalize_recording(recording_id, duration, file_size)

            if self.session_transcriber:
                self.session_transcriber.transcribe_recording(
                    recording_id=recording_id,
                    session_id=session_id,
                    participant_id=participant_id,
                    wav_filepath=filepath,
                    delete_after=True,
                    callback=self._on_transcription_done,
                )
            else:
                print("[Recorder] Transcriber not available — WAV file kept")
        elif recording_id:
            self.db.mark_recording_interrupted(recording_id)

        self.recording_id = None

    def _on_transcription_done(self, success: bool, text: str):
        if success:
            print(f"[Main] Transcription complete ({len(text)} chars)")
        else:
            print(f"[Main] Transcription failed: {text}")

    def _emergency_stop_recording(self):
        if self.session_recorder.is_recording:
            print("\n[Emergency] Saving recording before exit...")
            try:
                self._stop_recording()
            except Exception as e:
                print(f"[Emergency] Failed to save recording: {e}")

    def closeEvent(self, event):
        if self.session_recorder.is_recording:
            print("\n[Close] Saving recording before exit...")
            self._stop_recording()
        if self.current_session:
            self.on_session_stopped()
        event.accept()

    def _capture_forearm_image(self, stim_id: int):
        """
        Capture current forearm camera frame and save to captures/.
        Returns filepath or None.
        """
        if not self.forearm_camera:
            return None

        frame = self.forearm_camera.get_frame()
        if frame is None:
            return None

        import cv2
        save_dir = f"captures/forearm_{self.db_session_id:03d}"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"stim_{stim_id:04d}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"  Forearm image saved: {filepath}")
        return filepath

    # =====================================================================
    # Live forearm camera
    # =====================================================================

    def _update_forearm_feed(self):
        if hasattr(self, 'electrode_monitor') and self.electrode_monitor:
            self.electrode_monitor.update_positions(self.forearm_camera)
        if self.forearm_paused:
            return
        if self.forearm_camera and self.forearm_camera.is_running:
            frame = self.forearm_camera.get_frame()
            if frame is not None:
                self.body_view.update_live_frame(frame)

            state = self.forearm_camera.get_aruco_state()
            self._update_aruco_label(state)

            H = state.get("homography")
            if H is not None:
                self.body_view.update_aruco_grid(state)


    def _on_place_electrodes(self):
        """
        Validates mat + wrist + both electrodes are detected.
        Saves electrode positions to DB. Enables stimulate button.
        """
        print("\n=== Place Electrodes ===")

        self.body_view.clear_electrodes()
        self.body_view.clear_calibration()
        self.electrode_panel.reset()
        self._set_stimulate_enabled(False)

        if not self.forearm_camera:
            self._set_phase_status("No forearm camera active", "#dc2626")
            return

        state = self.forearm_camera.get_aruco_state()

        # ── Check 1: mat corners ──────────────────────────────────────────────
        if not (state.get("mat_ready") or state.get("ready")):
            self._set_phase_status(
                "Mat corners not detected — ensure IDs 0,2,3,4 are visible.", "#dc2626"
            )
            print("✗ Mat not ready")
            return

        # ── Check 2: wrist ArUco ──────────────────────────────────────────────
        wrist_px = state.get("wrist_center_px")
        wrist_mm = state.get("wrist_center_mm")
        if wrist_px is None or wrist_mm is None:
            self._set_phase_status(
                "Wrist marker (ID=1) not detected — place it on the wrist.", "#dc2626"
            )
            print("✗ Wrist ArUco not detected")
            return

        # ── Check 3: both electrode ArUcos ────────────────────────────────────
        electrodes = state.get("electrodes", {})
        missing = [eid for eid in settings.ELECTRODE_IDS if eid not in electrodes]
        if missing:
            self._set_phase_status(
                f"Electrode markers missing: ID {missing} — attach to electrode pads.",
                "#dc2626"
            )
            print(f"✗ Missing electrode ArUcos: {missing}")
            return

        # ── All detected — calibrate body view ───────────────────────────────
        ok = self.body_view.calibrate_from_aruco(state)
        if not ok:
            self._set_phase_status("Calibration failed — check marker visibility.", "#dc2626")
            return

        # ── Compute electrode positions relative to wrist ─────────────────────
        # Wrist = (0,0), toward elbow = positive along, thumb side = positive lateral
        arm_dir  = state.get("arm_dir")
        perp_dir = state.get("perp_dir")
        px_per_mm = state.get("px_per_mm", 1.0)

        electrode_data = []
        for eid, info in electrodes.items():
            e_px = info["pixel"]

            if arm_dir is not None and perp_dir is not None:
                import numpy as np
                relative   = e_px - wrist_px
                along_mm   = float(np.dot(relative, arm_dir))  / px_per_mm
                lateral_mm = float(np.dot(relative, perp_dir)) / px_per_mm
            else:
                along_mm   = info.get("down_mm",    0.0)
                lateral_mm = info.get("lateral_mm", 0.0)

            electrode_data.append({
                "electrode_id": eid,
                "pixel_x":      float(e_px[0]),
                "pixel_y":      float(e_px[1]),
                "along_mm":     round(along_mm,   1),   # toward elbow
                "lateral_mm":   round(lateral_mm, 1),   # thumb=+, pinky=-
                "wrist_mat_x":  round(float(wrist_mm[0]), 1),  # wrist in mat frame
                "wrist_mat_y":  round(float(wrist_mm[1]), 1),
            })

            print(f"  E{eid}: {along_mm:.1f}mm from wrist toward elbow, "
                f"{lateral_mm:.1f}mm lateral (+ = thumb)")

        print(f"  Wrist in mat frame: "
            f"({wrist_mm[0]:.1f}, {wrist_mm[1]:.1f}) mm")

        # Store for use when stimulate is pressed
        self._electrode_placement = electrode_data
        self._wrist_mat_mm        = wrist_mm


        # ── Update UI ─────────────────────────────────────────────────────────
        placed = self.body_view.auto_place_electrodes_from_aruco(state)
        self.electrode_panel.set_confirmed(len(electrode_data))
        self._set_stimulate_enabled(True)
        self._set_phase_status(
            f"✓ {len(electrode_data)} electrodes detected and saved. Ready to stimulate!",
            "#16a34a"
        )
        print(f"✓ Electrode placement saved — stimulation enabled")


    def _resume_forearm_feed(self):
        """Resume live feed."""
        self.forearm_paused = False
        self.body_view._placement_enabled = False
        print("Live feed active")

    def _on_calibration_complete(self, mapper):
        """Called after ArUco calibration."""
        pass


# =========================================================================
# Adapter: bridges split tracker results → database format
# =========================================================================

class _CombinedMovementResult:
    """
    Bridges separate WristMovementResult + HandMovementResult into the
    combined format that db.record_movement() expects.
    """

    def __init__(self, wrist_result=None, finger_result=None):
        # Wrist data (from PoseTracker)
        if wrist_result and wrist_result.pose_detected:
            self.wrist_angle_delta = wrist_result.wrist_angle_delta
            self.wrist_direction   = wrist_result.wrist_direction
        else:
            self.wrist_angle_delta = 0.0
            self.wrist_direction   = "no_data"

        # Finger data (from HandTracker)
        if finger_result and finger_result.hand_detected:
            self.total_finger_movement   = finger_result.total_finger_movement
            self.primary_finger          = finger_result.primary_finger
            self.primary_finger_movement = finger_result.primary_finger_movement
            self.movement_type           = finger_result.movement_type
            self.flexion_change          = finger_result.flexion_change
            self.baseline_finger_angles  = finger_result.baseline_finger_angles
            self.result_finger_angles    = finger_result.result_finger_angles
            self.finger_angle_deltas     = finger_result.finger_angle_deltas
            self.latency_ms              = finger_result.latency_ms
            self.confidence              = finger_result.confidence
            self.hand_detected           = True
        else:
            self.total_finger_movement   = 0.0
            self.primary_finger          = ""
            self.primary_finger_movement = 0.0
            self.movement_type           = "no_data"
            self.flexion_change          = {
                "thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0
            }
            self.baseline_finger_angles  = {}
            self.result_finger_angles    = {}
            self.finger_angle_deltas     = {}
            self.confidence              = 0.0
            self.hand_detected           = False
            if wrist_result and wrist_result.pose_detected:
                self.latency_ms = wrist_result.latency_ms
            else:
                self.latency_ms = 0.0










#----------_-----_Electrode Position Monitoring-----------------------__------_---__-______--_-

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer

class ElectrodePositionMonitor(QDialog):
    """Live monitor showing electrode positions relative to wrist."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Electrode Position Monitor")
        self.setMinimumWidth(400)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.labels = {}
        for eid in [5, 6]:
            lbl = QLabel(f"E{eid}: waiting...")
            lbl.setStyleSheet("font-family: monospace; font-size: 13px; padding: 8px;")
            layout.addWidget(lbl)
            self.labels[eid] = lbl

        self.wrist_lbl = QLabel("Wrist: waiting...")
        self.wrist_lbl.setStyleSheet("font-family: monospace; font-size: 13px; padding: 8px; color: #2563eb;")
        layout.addWidget(self.wrist_lbl)

    def update_positions(self, forearm_camera):
        """Call this every frame to update displayed positions."""
        if not forearm_camera:
            return

        state      = forearm_camera.get_aruco_state()
        H          = state.get("homography")
        wrist_px   = state.get("wrist_center_px")
        electrodes = state.get("electrodes", {})

        if H is None or wrist_px is None:
            return

        import cv2, numpy as np

        def px_to_mat_mm(px_pt):
            pt     = np.array([[[float(px_pt[0]), float(px_pt[1])]]], dtype=np.float32)
            result = cv2.perspectiveTransform(pt, H)
            return np.array([result[0][0][0], result[0][0][1]], dtype=np.float32)

        wrist_mm = px_to_mat_mm(wrist_px)
        self.wrist_lbl.setText(
            f"Wrist: px=({wrist_px[0]:.0f}, {wrist_px[1]:.0f})  "
            f"mat=({wrist_mm[0]:.1f}, {wrist_mm[1]:.1f}) mm"
        )

        for eid, info in electrodes.items():
            if eid not in self.labels:
                continue
            e_px    = info["pixel"]
            e_mm    = px_to_mat_mm(e_px)
            diff_mm = e_mm - wrist_mm
            dx_mm   = float(diff_mm[0])
            dy_mm   = float(diff_mm[1])
            dist_mm = float(np.linalg.norm(diff_mm))

            self.labels[eid].setText(
                f"E{eid}:  px=({e_px[0]:.0f}, {e_px[1]:.0f})\n"
                f"       mat=({e_mm[0]:.1f}, {e_mm[1]:.1f}) mm\n"
                f"       from wrist: dx={dx_mm:+.1f}mm  dy={dy_mm:+.1f}mm  dist={dist_mm:.1f}mm"
            )