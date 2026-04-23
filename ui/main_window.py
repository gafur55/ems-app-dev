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

# =====================================================================
# CAMERA CONFIGURATION — change these to match your setup
# =====================================================================
# Run `python3 find_cameras.py` to see available cameras and their indices.
#
# 3-camera setup (2 iPhones + laptop):
#   HAND_CAMERA    = 0   # iPhone #1 (Continuity) → finger tracking
#   POSE_CAMERA    = 1   # Laptop webcam           → wrist tracking
#   FOREARM_CAMERA = 2   # iPhone #2 (Camo)        → forearm live view
#
# Laptop-only setup (no iPhones):
#   HAND_CAMERA    = 0   # Laptop webcam → finger tracking
#   POSE_CAMERA    = 0   # Laptop webcam → wrist tracking (same camera)
#   FOREARM_CAMERA = -1  # Disabled (use static image)
#
HAND_CAMERA    = 0
POSE_CAMERA    = 2
FOREARM_CAMERA = 3

ARM_SIDE = "left" #mirror effect (left here is right in reality)


class EMSWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.current_session = None
        self.ems_controller = None

        # Split trackers
        self.pose_tracker = None   # laptop cam → wrist angle
        self.hand_tracker = None   # phone cam  → finger angles
        self.pose_preview = None
        self.hand_preview = None

        # Forearm camera (live view for electrode placement)
        self.forearm_camera = None
        self.forearm_timer = None  # QTimer for updating live feed
        self.forearm_paused = False

        # Audio recorder
        self.session_recorder = SessionRecorder(save_dir="data/recordings")
        self.session_transcriber = None  # initialized after sign-in when db is available
        self.recording_id = None  # DB recording_id

        # Register atexit handler to save recording on unexpected exit
        import atexit
        atexit.register(self._emergency_stop_recording)

        # Database — auto-creates data/ems_data.db
        self.db = EMSDatabase()
        self.db_session_id = None

        # Participant data (loaded from sign-in)
        self.participant = None
        self.forearm_length_cm = None

        self.setup_ui()

        # Show sign-in dialog on startup
        self._show_sign_in()

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
        """Create and configure the body diagram view"""
        self.body_view = BodyDiagramView()
        self.body_view.setMinimumWidth(500)
        parent_layout.addWidget(self.body_view, stretch=1)

        # Calibration is ArUco-only — signal wired for status update
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

        # Single button: Place Electrodes → pause + clear + calibrate
        cc = self.parameter_panel.capture_controls
        cc.place_electrodes_requested.connect(self._on_place_electrodes)

        parent_layout.addWidget(self.parameter_panel)

    def _setup_stimulation_panel(self, parent_layout):
        """Create and configure the stimulation panel"""
        self.stimulation_panel = StimulationPanel()
        self.stimulation_panel.stimulate_requested.connect(self.on_stimulate_requested)
        parent_layout.addWidget(self.stimulation_panel)

        # Stimulate button starts disabled — must place electrodes first
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
        # Stop any active session first
        if self.current_session:
            self.on_session_stopped()

        dialog = SignInDialog(self.db, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            self.participant = dialog.get_participant_data()
            pid = self.participant["participant_id"]

            # Update header bar
            name = self.participant.get("name") or pid
            parts = [f"{name} ({pid})"]
            if self.participant.get("age"):
                parts.append(f"Age: {self.participant['age']}")
            if self.participant.get("experience_level"):
                parts.append(f"Exp: {self.participant['experience_level']}")
            self.participant_label.setText("  |  ".join(parts))

            print(f"\n✓ Signed in as {pid}")
        else:
            # If no participant was ever set (first launch, user cancelled)
            if self.participant is None:
                self.participant_label.setText("No participant — click 'Switch Participant'")

    # =====================================================================
    # Session lifecycle
    # =====================================================================

    def on_session_started(self, params):
        """
        Handle session start event.
        Uses participant from sign-in + device from session form.
        """
        # Check participant is signed in
        if not self.participant:
            print("✗ No participant signed in — showing sign-in dialog")
            self._show_sign_in()
            if not self.participant:
                return

        pid = self.participant["participant_id"]
        device_name = params["device_name"]

        # Create session object
        self.current_session = Session(
            participant_id=pid,
            device_name=device_name
        )
        self.current_session.age = self.participant.get("age")
        self.current_session.arm_width = self.participant.get("arm_width")
        self.current_session.arm_length = self.participant.get("arm_length")
        self.current_session.skin_resistance_100hz_kohm = self.participant.get("skin_resistance_100hz_kohm")
        self.current_session.skin_resistance_1khz_kohm = self.participant.get("skin_resistance_1khz_kohm")
        self.current_session.skin_resistance_10khz_kohm = self.participant.get("skin_resistance_10khz_kohm")
        self.current_session.skin_resistance_100khz_kohm = self.participant.get("skin_resistance_100khz_kohm")
        self.current_session.pain_threshold_ma = self.participant.get("pain_threshold_ma")
        self.current_session.experience_level = self.participant.get("experience_level")

        print(f"\n=== Session Created ===")
        print(f"Participant: {pid}")
        print(f"Device: {device_name}")

        # Save session to database (participant already in DB from sign-in)
        self.db_session_id = self.db.create_session(
            participant_id=pid,
            device_name=device_name,
            forearm_length_cm=self.participant.get("arm_length"),
        )

        # Connect to EMS device
        print(f"\n=== Connecting to EMS Device ===")

        # Initialize transcriber now that db is available
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

            # Start both trackers
            self._start_trackers()

            # Start audio recording
            self._start_recording()

            # Enable electrode placement (stimulate stays disabled)
            self.electrode_panel.setEnabled(True)
            self._set_stimulate_enabled(False)

            self._set_phase_status("Click 'Place Electrodes' to begin", "#FF9800")
            print(f"Ready to place electrodes!")
        else:
            print("✗ Failed to connect to device")
            print("  Check that device is plugged in and try again")

    def on_session_stopped(self):
        """
        Handle session stop.
        Stop trackers, disconnect device, end DB session, reset UI.
        """
        print("\n=== Stopping Session ===")

        # Stop audio recording first (before anything else)
        self._stop_recording()

        # Stop trackers
        self._stop_trackers()

        # Disconnect EMS device
        if self.ems_controller and self.ems_controller.is_connected():
            self.ems_controller.disconnect()
            print("✓ Device disconnected")

        # Clear electrodes
        self.body_view.clear_electrodes()
        print("✓ Electrodes cleared")

        # ---- End session in database ----
        if self.db_session_id:
            # Auto-save pain threshold = max intensity used in this session
            max_intensity = self.db.get_session_max_intensity(self.db_session_id)
            if max_intensity is not None and self.participant:
                pid = self.participant["participant_id"]
                self.db.update_pain_threshold_ma(pid, max_intensity)
                print(f"  Pain threshold updated: {max_intensity} mA (max intensity used)")

            self.db.end_session(self.db_session_id)
            self.db.print_stats()
            self.db_session_id = None

        # Disable controls
        self.body_view.setEnabled(False)
        self.stimulation_panel.setEnabled(False)
        self._set_stimulate_enabled(False)
        self.electrode_panel.reset()
        self.electrode_panel.setEnabled(False)

        # Reset session
        self.current_session = None
        self.ems_controller = None

        print("✓ Session stopped - ready to start new session")

    # =====================================================================
    # Single status line — shown in parameter_panel.status_label
    # =====================================================================

    def _set_phase_status(self, message: str, color: str = "gray") -> None:
        """Update the single status label shown on the right panel."""
        self.parameter_panel.status_label.setText(message)
        self.parameter_panel.status_label.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 13px;"
        )

    # =====================================================================
    # Electrode placement gate
    # =====================================================================

    def _set_stimulate_enabled(self, enabled: bool) -> None:
        """Enable/disable the STIMULATE button."""
        self.stimulation_panel.stimulate_button.setEnabled(enabled)

    def on_electrodes_confirmed(self):
        """
        Handle 'Confirm Electrode Placement' button click.
        Validates at least 2 electrodes are placed, then enables STIMULATE.
        """
        electrode_count = len(self.body_view.electrodes)

        if electrode_count < 2:
            print(f"✗ Need at least 2 electrodes (anode + cathode), currently: {electrode_count}")
            self.electrode_panel.set_error(
                f"Place at least 2 electrodes first ({electrode_count} placed)"
            )
            self._set_phase_status(f"Place at least 2 electrodes ({electrode_count} placed)", "#dc2626")
            return

        print(f"\n=== Electrodes Confirmed ({electrode_count} placed) ===")

        # Enable stimulation
        self._set_stimulate_enabled(True)
        self.electrode_panel.set_confirmed(electrode_count)

        self._set_phase_status("Ready to stimulate!", "#16a34a")

        # Resume the live forearm feed
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
        channel = params["channel"]
        intensity = params["intensity"]
        pulse_width = params["pulse_width"]
        pulse_count = params["pulse_count"]
        delay = params["delay"]

        print(f"\n=== Stimulation Requested ===")
        print(f"Channel: {channel}")
        print(f"Intensity: {intensity} mA")
        print(f"Pulse Width: {pulse_width} μs")
        print(f"Pulse Count: {pulse_count}")
        print(f"Delay: {delay} ms")

        # Check device
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

            # Save electrode positions directly from live ArUco state
            if self.forearm_camera:
                aruco_state = self.forearm_camera.get_aruco_state()
                electrodes  = aruco_state.get("electrodes", {})
                for channel, (eid, info) in enumerate(electrodes.items()):
                    self.db.record_electrode(
                        session_id=self.db_session_id,
                        channel=channel,
                        pixel_x=float(info["pixel"][0]),
                        pixel_y=float(info["pixel"][1]),
                        stim_id=stim_id,
                        real_x_cm=info["lateral_mm"] / 10.0,  # mm → cm, thumb = positive
                        real_y_cm=info["down_mm"]    / 10.0,  # mm → cm, toward wrist
                    )

        # ---- STEP 3: Start recording + stimulate simultaneously ----
        stim_duration_s = (pulse_count * delay) / 1000.0
        recording_duration = stim_duration_s + 1.5  # 1.5s tail to capture peak after stim

        # Results containers for threads
        wrist_result_box = [None]
        finger_result_box = [None]

        # Start pose recording thread (wrist angle)
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

        # Start hand recording thread (finger angles)
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

        # Small delay to let recording threads start capturing
        time.sleep(0.05)

        # Fire stimulation (blocks while pulses are sent)
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
        wrist_result = None
        finger_result = None

        if pose_thread:
            pose_thread.join(timeout=5.0)
            wrist_result = wrist_result_box[0]

        if hand_thread:
            hand_thread.join(timeout=5.0)
            finger_result = finger_result_box[0]

        # Build status message
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

        # Save combined movement result to database
        if stim_id:
            combined = _CombinedMovementResult(wrist_result, finger_result)
            print(f"\n  DB DEBUG: wrist_delta={combined.wrist_angle_delta}, "
                  f"wrist_dir={combined.wrist_direction}, "
                  f"finger_total={combined.total_finger_movement}, "
                  f"hand_detected={combined.hand_detected}")
            self.db.record_movement(stim_id, combined)

    # =====================================================================
    # Tracker management (2 cameras)
    # =====================================================================

    def _start_trackers(self) -> None:
        """
        Start both movement trackers and open preview windows.

        Camera assignments:
            camera 0 = iPhone #1      → HandTracker    (finger angles)
            camera 1 = laptop webcam   → PoseTracker    (wrist angle)
            camera 2 = iPhone #2       → ForearmCamera  (live forearm view)

        Adjust camera_index values if your setup differs.
        """
        arm_side = ARM_SIDE   # auto-detects most visible arm

        print("\n=== Starting Movement Trackers ===")

        # --- Forearm camera: live forearm view ---
        if FOREARM_CAMERA >= 0:
            self.forearm_camera = ForearmCamera(camera_index=FOREARM_CAMERA)
            if self.forearm_camera.start():
                print("✓ Forearm camera ready (live view)")
                # Start QTimer for live feed updates (~30 FPS)
                self.forearm_timer = QTimer()
                self.forearm_timer.timeout.connect(self._update_forearm_feed)
                self.forearm_timer.start(33)  # ~30 FPS
                self.forearm_paused = False
                # Enable pause button
                self.parameter_panel.capture_controls.set_camera_active(True)
            else:
                print("⚠ Forearm camera failed to start")
                self.forearm_camera = None
        else:
            print("  Forearm camera disabled (FOREARM_CAMERA = -1)")
            self.forearm_camera = None

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
                f"Wrist Tracking — Pose (auto arm) [Laptop Camera]"
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
        # Stop forearm feed
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

    # =====================================================================
    # Audio recording
    # =====================================================================

    def _start_recording(self):
        """Start audio recording for the current session."""
        if not self.db_session_id or not self.participant:
            return

        pid = self.participant["participant_id"]
        filepath = self.session_recorder.start(
            session_id=self.db_session_id,
            participant_id=pid,
        )

        if filepath:
            # Register in DB immediately (status='recording')
            self.recording_id = self.db.start_recording(
                session_id=self.db_session_id,
                participant_id=pid,
                filepath=filepath,
            )

    def _stop_recording(self):
        """Stop audio recording, finalize in DB, and trigger transcription."""
        if not self.session_recorder.is_recording:
            return

        filepath = self.session_recorder.stop()

        # Finalize recording in DB
        recording_id = self.recording_id
        session_id = self.db_session_id
        participant_id = self.participant["participant_id"] if self.participant else ""

        if recording_id and filepath and os.path.exists(filepath):
            duration = self.session_recorder._frames_written / self.session_recorder.SAMPLE_RATE
            file_size = os.path.getsize(filepath)
            self.db.finalize_recording(recording_id, duration, file_size)

            # Trigger background transcription → deletes WAV after success
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
        """Callback when background transcription finishes."""
        if success:
            print(f"[Main] Transcription complete ({len(text)} chars)")
        else:
            print(f"[Main] Transcription failed: {text}")

    def _emergency_stop_recording(self):
        """
        atexit handler: save recording if app exits unexpectedly.
        Called by Python on interpreter shutdown.
        """
        if self.session_recorder.is_recording:
            print("\n[Emergency] Saving recording before exit...")
            try:
                self._stop_recording()
            except Exception as e:
                print(f"[Emergency] Failed to save recording: {e}")

    def closeEvent(self, event):
        """
        Handle window close (X button, Cmd+Q, etc).
        Ensures recording is saved before the app exits.
        """
        # Stop recording first
        if self.session_recorder.is_recording:
            print("\n[Close] Saving recording before exit...")
            self._stop_recording()

        # Stop session cleanly if active
        if self.current_session:
            self.on_session_stopped()

        event.accept()

    # =====================================================================
    # Live forearm camera: pause/resume/calibrate
    # =====================================================================

    def _update_forearm_feed(self):
        """QTimer callback: push latest camera frame to body view."""
        if self.forearm_paused:
            return
        if self.forearm_camera and self.forearm_camera.is_running:
            frame = self.forearm_camera.get_frame()
            if frame is not None:
                self.body_view.update_live_frame(frame)

    def _on_place_electrodes(self):
        """
        ArUco-only electrode placement flow.

        The forearm camera runs ArUco detection continuously.
        When this button is pressed:
          - If all 4 reference markers are detected → calibrate instantly
          - If electrode markers (IDs 4, 5) are also detected → auto-place them
          - Otherwise → user clicks manually to place electrodes
        No feed pause, no manual calibration clicks needed.
        """
        print("\n=== Place Electrodes ===")

        # Clear previous state
        self.body_view.clear_electrodes()
        self.body_view.clear_calibration()
        self.electrode_panel.reset()
        self._set_stimulate_enabled(False)

        # Check ArUco is ready
        if not self.forearm_camera or not self.forearm_camera.is_aruco_ready():
            self._set_phase_status(
                "ArUco markers not detected — ensure all 4 corner markers are visible.",
                "#dc2626"
            )
            print("✗ ArUco not ready — hold all 4 forearm markers in view and try again.")
            return

        # Grab latest ArUco state
        aruco_state = self.forearm_camera.get_aruco_state()

        # Calibrate from ArUco
        self._set_phase_status("Calibrating from ArUco markers...", "#FF9800")
        ok = self.body_view.calibrate_from_aruco(aruco_state)
        if not ok:
            self._set_phase_status("Calibration failed — check marker visibility.", "#dc2626")
            return

        # Auto-place electrode markers if IDs 4/5 are detected
        placed = self.body_view.auto_place_electrodes_from_aruco(aruco_state)

        if placed > 0:
            self._set_phase_status(
                f"Calibrated. {placed} electrode(s) auto-placed. Confirm when ready.",
                "#2563eb"
            )
        else:
            self._set_phase_status(
                "Calibrated via ArUco. Click on image to place electrodes, then Confirm.",
                "#2563eb"
            )

    def _resume_forearm_feed(self):
        """Resume live feed (kept for on_electrodes_confirmed compatibility)."""
        self.forearm_paused = False
        self.body_view._placement_enabled = False
        print("Live feed active")

    def _on_calibration_complete(self, mapper):
        """Called after ArUco calibration — status already set in _on_place_electrodes."""
        pass


# =========================================================================
# Adapter: bridges split tracker results → database format
# =========================================================================

class _CombinedMovementResult:
    """
    Bridges separate WristMovementResult + HandMovementResult into the
    combined format that db.record_movement() expects.

    db.record_movement() reads:
        wrist_angle_delta, wrist_direction,
        total_finger_movement, primary_finger, primary_finger_movement,
        movement_type,
        flexion_change (dict: finger → total),
        baseline_finger_angles, result_finger_angles, finger_angle_deltas (dicts),
        latency_ms, confidence, hand_detected
    """

    def __init__(self, wrist_result=None, finger_result=None):
        # --- Wrist data (from PoseTracker) ---
        if wrist_result and wrist_result.pose_detected:
            self.wrist_angle_delta = wrist_result.wrist_angle_delta
            self.wrist_direction = wrist_result.wrist_direction
        else:
            self.wrist_angle_delta = 0.0
            self.wrist_direction = "no_data"

        # --- Finger data (from HandTracker) ---
        if finger_result and finger_result.hand_detected:
            self.total_finger_movement = finger_result.total_finger_movement
            self.primary_finger = finger_result.primary_finger
            self.primary_finger_movement = finger_result.primary_finger_movement
            self.movement_type = finger_result.movement_type
            self.flexion_change = finger_result.flexion_change
            self.baseline_finger_angles = finger_result.baseline_finger_angles
            self.result_finger_angles = finger_result.result_finger_angles
            self.finger_angle_deltas = finger_result.finger_angle_deltas
            self.latency_ms = finger_result.latency_ms
            self.confidence = finger_result.confidence
            self.hand_detected = True
        else:
            self.total_finger_movement = 0.0
            self.primary_finger = ""
            self.primary_finger_movement = 0.0
            self.movement_type = "no_data"
            self.flexion_change = {
                "thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0
            }
            self.baseline_finger_angles = {}
            self.result_finger_angles = {}
            self.finger_angle_deltas = {}
            self.confidence = 0.0
            self.hand_detected = False
            # Fall back to wrist latency
            if wrist_result and wrist_result.pose_detected:
                self.latency_ms = wrist_result.latency_ms
            else:
                self.latency_ms = 0.0