"""
Participant sign-in dialog.

Shown at application startup before the main experiment GUI.
Allows selecting an existing participant (auto-loads their info)
or creating a new one (fills in once, saved to DB).

No passwords — just identity selection for data collection.

Emits:
    None — call get_participant_data() after exec() returns Accepted.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QGroupBox, QFrame, QStackedWidget, QWidget
)
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QFont
from PyQt6.QtCore import Qt
from typing import Optional, Dict


class SignInDialog(QDialog):
    """
    Participant sign-in dialog.

    Usage:
        dialog = SignInDialog(db)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            participant = dialog.get_participant_data()
            # participant = {"participant_id": "P001", "age": 25, ...}
    """

    def __init__(self, db, parent=None):
        """
        Args:
            db: EMSDatabase instance.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.db = db
        self._participant_data: Optional[Dict] = None

        self.setWindowTitle("EMS Lab — Participant Sign In")
        self.setMinimumWidth(500)
        self.setModal(True)

        self._setup_ui()
        self._load_participants()

    # ==================================================================
    # UI Setup
    # ==================================================================

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)
        self.setLayout(layout)

        # --- Header ---
        header = QLabel("Participant Sign In")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #80000; padding: 10px;")
        layout.addWidget(header)

        # --- Stacked widget: existing / new ---
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Page 0: Select existing participant
        self._build_existing_page()

        # Page 1: New participant form
        self._build_new_page()

        self.stack.setCurrentIndex(0)

    def _build_existing_page(self):
        """Page for selecting an existing participant."""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        page.setLayout(layout)

        # Dropdown
        layout.addWidget(QLabel("Select participant:"))
        self.participant_combo = QComboBox()
        self.participant_combo.setMinimumHeight(36)
        self.participant_combo.setStyleSheet("font-size: 14px; padding: 4px;")
        self.participant_combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self.participant_combo)

        # Info preview
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "color: #4b5563; font-size: 12px; padding: 8px; "
            "background-color: #f3f4f6; border-radius: 4px;"
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.new_btn = QPushButton("+ New Participant")
        self.new_btn.setMinimumHeight(40)
        self.new_btn.setStyleSheet("""
            QPushButton {
                background-color: #800000;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
        """)
        self.new_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_layout.addWidget(self.new_btn)

        self.continue_btn = QPushButton("Continue →")
        self.continue_btn.setMinimumHeight(40)
        self.continue_btn.setEnabled(False)
        self.continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover { background-color: #15803d; }
            QPushButton:disabled { background-color: #9ca3af; }
        """)
        self.continue_btn.clicked.connect(self._on_existing_continue)
        btn_layout.addWidget(self.continue_btn)

        layout.addLayout(btn_layout)
        self.stack.addWidget(page)

    def _build_new_page(self):
        """Page for creating a new participant."""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        page.setLayout(layout)

        # Back button
        back_btn = QPushButton("← Back to selection")
        back_btn.setStyleSheet(
            "border: none; color: #2563eb; font-size: 12px; text-align: left;"
        )
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(back_btn)

        # Form
        group = QGroupBox("New Participant")
        form = QGridLayout()
        form.setSpacing(8)
        group.setLayout(form)

        # Row 0: Name
        form.addWidget(QLabel("Name *"), 0, 0)
        self.new_name_input = QLineEdit()
        self.new_name_input.setPlaceholderText("e.g., John Smith")
        self.new_name_input.setMinimumHeight(32)
        form.addWidget(self.new_name_input, 0, 1)

        # Row 1: Participant ID
        form.addWidget(QLabel("Participant ID *"), 1, 0)
        self.new_id_input = QLineEdit()
        self.new_id_input.setPlaceholderText("e.g., P001")
        self.new_id_input.setMinimumHeight(32)
        form.addWidget(self.new_id_input, 1, 1)

        # Row 2: Age
        form.addWidget(QLabel("Age"), 2, 0)
        self.new_age_input = QLineEdit()
        self.new_age_input.setPlaceholderText("years")
        self.new_age_input.setValidator(QIntValidator(0, 120))
        form.addWidget(self.new_age_input, 2, 1)

        # Row 3: Arm Width
        form.addWidget(QLabel("Arm Width (cm)"), 3, 0)
        self.new_arm_width_input = QLineEdit()
        self.new_arm_width_input.setPlaceholderText("cm")
        self.new_arm_width_input.setValidator(QDoubleValidator(0.0, 100.0, 1))
        form.addWidget(self.new_arm_width_input, 3, 1)

        # Row 4: Arm Length
        form.addWidget(QLabel("Arm Length (cm)"), 4, 0)
        self.new_arm_length_input = QLineEdit()
        self.new_arm_length_input.setPlaceholderText("wrist to elbow, cm")
        self.new_arm_length_input.setValidator(QDoubleValidator(0.0, 100.0, 1))
        form.addWidget(self.new_arm_length_input, 4, 1)

        # Row 5: Skin Resistance (4 frequencies)
        resistance_label = QLabel("Skin Resistance (kΩ)")
        resistance_label.setStyleSheet("font-weight: bold;")
        form.addWidget(resistance_label, 5, 0, 1, 2)

        freq_layout = QGridLayout()
        freq_layout.setSpacing(6)

        freq_layout.addWidget(QLabel("100 Hz"), 0, 0)
        self.resistance_100hz = QLineEdit()
        self.resistance_100hz.setPlaceholderText("kΩ")
        self.resistance_100hz.setValidator(QDoubleValidator(0.0, 99999.0, 2))
        freq_layout.addWidget(self.resistance_100hz, 0, 1)

        freq_layout.addWidget(QLabel("1 kHz"), 0, 2)
        self.resistance_1khz = QLineEdit()
        self.resistance_1khz.setPlaceholderText("kΩ")
        self.resistance_1khz.setValidator(QDoubleValidator(0.0, 99999.0, 2))
        freq_layout.addWidget(self.resistance_1khz, 0, 3)

        freq_layout.addWidget(QLabel("10 kHz"), 1, 0)
        self.resistance_10khz = QLineEdit()
        self.resistance_10khz.setPlaceholderText("kΩ")
        self.resistance_10khz.setValidator(QDoubleValidator(0.0, 99999.0, 2))
        freq_layout.addWidget(self.resistance_10khz, 1, 1)

        freq_layout.addWidget(QLabel("100 kHz"), 1, 2)
        self.resistance_100khz = QLineEdit()
        self.resistance_100khz.setPlaceholderText("kΩ")
        self.resistance_100khz.setValidator(QDoubleValidator(0.0, 99999.0, 2))
        freq_layout.addWidget(self.resistance_100khz, 1, 3)

        form.addLayout(freq_layout, 6, 0, 1, 2)

        # Row 7: Experience Level
        form.addWidget(QLabel("Experience Level"), 7, 0)
        self.new_experience_combo = QComboBox()
        self.new_experience_combo.addItems(["novice", "intermediate", "expert"])
        form.addWidget(self.new_experience_combo, 7, 1)

        layout.addWidget(group)

        # Error label
        self.new_error_label = QLabel("")
        self.new_error_label.setStyleSheet("color: #dc2626; font-size: 12px;")
        layout.addWidget(self.new_error_label)

        # Create button
        create_btn = QPushButton("Create & Continue →")
        create_btn.setMinimumHeight(40)
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover { background-color: #15803d; }
        """)
        create_btn.clicked.connect(self._on_create_new)
        layout.addWidget(create_btn)

        self.stack.addWidget(page)

    # ==================================================================
    # Logic
    # ==================================================================

    def _load_participants(self):
        """Load participants from DB into the dropdown."""
        self.participant_combo.clear()
        self.participant_combo.addItem("— Select participant —", None)

        participants = self.db.get_all_participants()
        for p in participants:
            name = p.get("name") or p["participant_id"]
            label = f"{name} ({p['participant_id']})"
            if p.get("session_count"):
                label += f" — {p['session_count']} sessions"
            self.participant_combo.addItem(label, p["participant_id"])

        self.continue_btn.setEnabled(False)
        self.info_label.setText("Select a participant or create a new one.")

    def _on_selection_changed(self, index):
        """Update info preview when dropdown selection changes."""
        pid = self.participant_combo.currentData()
        if pid is None:
            self.continue_btn.setEnabled(False)
            self.info_label.setText("Select a participant or create a new one.")
            return

        p = self.db.get_participant(pid)
        if p is None:
            return

        self.continue_btn.setEnabled(True)

        # Build info string
        info_parts = [f"ID: {p['participant_id']}"]
        if p.get("name"):
            info_parts.insert(0, f"Name: {p['name']}")
        if p.get("age"):
            info_parts.append(f"Age: {p['age']}")
        if p.get("arm_width_cm"):
            info_parts.append(f"Arm Width: {p['arm_width_cm']} cm")
        if p.get("arm_length_cm"):
            info_parts.append(f"Arm Length: {p['arm_length_cm']} cm")
        # Skin resistance
        r_parts = []
        if p.get("skin_resistance_100hz_kohm"):
            r_parts.append(f"100Hz: {p['skin_resistance_100hz_kohm']}kΩ")
        if p.get("skin_resistance_1khz_kohm"):
            r_parts.append(f"1kHz: {p['skin_resistance_1khz_kohm']}kΩ")
        if p.get("skin_resistance_10khz_kohm"):
            r_parts.append(f"10kHz: {p['skin_resistance_10khz_kohm']}kΩ")
        if p.get("skin_resistance_100khz_kohm"):
            r_parts.append(f"100kHz: {p['skin_resistance_100khz_kohm']}kΩ")
        if r_parts:
            info_parts.append("R: " + ", ".join(r_parts))
        if p.get("pain_threshold_ma"):
            info_parts.append(f"Pain Threshold: {p['pain_threshold_ma']} mA")
        if p.get("experience_level"):
            info_parts.append(f"Experience: {p['experience_level']}")

        self.info_label.setText("  |  ".join(info_parts))

    def _on_existing_continue(self):
        """Continue with selected existing participant."""
        pid = self.participant_combo.currentData()
        if pid is None:
            return

        p = self.db.get_participant(pid)
        if p is None:
            return

        self._participant_data = {
            "participant_id": p["participant_id"],
            "name": p.get("name"),
            "age": p.get("age"),
            "arm_width": p.get("arm_width_cm"),
            "arm_length": p.get("arm_length_cm"),
            "skin_resistance_100hz_kohm": p.get("skin_resistance_100hz_kohm"),
            "skin_resistance_1khz_kohm": p.get("skin_resistance_1khz_kohm"),
            "skin_resistance_10khz_kohm": p.get("skin_resistance_10khz_kohm"),
            "skin_resistance_100khz_kohm": p.get("skin_resistance_100khz_kohm"),
            "pain_threshold_ma": p.get("pain_threshold_ma"),
            "experience_level": p.get("experience_level"),
        }

        display = p.get("name") or p["participant_id"]
        print(f"✓ Signed in as: {display} ({p['participant_id']})")
        self.accept()

    def _on_create_new(self):
        """Validate and create a new participant."""
        name = self.new_name_input.text().strip()
        pid = self.new_id_input.text().strip()

        if not name:
            self.new_error_label.setText("Name is required.")
            return

        if not pid:
            self.new_error_label.setText("Participant ID is required.")
            return

        # Check if already exists
        existing = self.db.get_participant(pid)
        if existing:
            self.new_error_label.setText(
                f"'{pid}' already exists. Go back and select from dropdown."
            )
            return

        # Gather fields
        age = self._safe_int(self.new_age_input.text())
        arm_width = self._safe_float(self.new_arm_width_input.text())
        arm_length = self._safe_float(self.new_arm_length_input.text())
        r_100hz = self._safe_float(self.resistance_100hz.text())
        r_1khz = self._safe_float(self.resistance_1khz.text())
        r_10khz = self._safe_float(self.resistance_10khz.text())
        r_100khz = self._safe_float(self.resistance_100khz.text())
        experience = self.new_experience_combo.currentText()

        # Save to database
        self.db.create_participant(
            participant_id=pid,
            name=name,
            age=age,
            arm_width_cm=arm_width,
            arm_length_cm=arm_length,
            skin_resistance_100hz_kohm=r_100hz,
            skin_resistance_1khz_kohm=r_1khz,
            skin_resistance_10khz_kohm=r_10khz,
            skin_resistance_100khz_kohm=r_100khz,
            experience_level=experience,
        )

        self._participant_data = {
            "participant_id": pid,
            "name": name,
            "age": age,
            "arm_width": arm_width,
            "arm_length": arm_length,
            "skin_resistance_100hz_kohm": r_100hz,
            "skin_resistance_1khz_kohm": r_1khz,
            "skin_resistance_10khz_kohm": r_10khz,
            "skin_resistance_100khz_kohm": r_100khz,
            "experience_level": experience,
        }

        print(f"✓ New participant created: {name} ({pid})")
        self.accept()

    # ==================================================================
    # Public API
    # ==================================================================

    def get_participant_data(self) -> Optional[Dict]:
        """
        Get the selected/created participant data.

        Returns:
            Dict with keys: participant_id, age, arm_width, arm_length,
                            skin_impedance, pain_threshold, experience_level.
            None if dialog was cancelled.
        """
        return self._participant_data

    # ==================================================================
    # Helpers
    # ==================================================================

    def _safe_int(self, text: str) -> Optional[int]:
        try:
            return int(text) if text.strip() else None
        except ValueError:
            return None

    def _safe_float(self, text: str) -> Optional[float]:
        try:
            return float(text) if text.strip() else None
        except ValueError:
            return None