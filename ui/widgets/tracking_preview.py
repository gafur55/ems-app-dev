"""
Hand tracking preview window for the EMS GUI.

Displays the live webcam feed with hand skeleton overlay
by pulling annotated frames from MovementTracker via a QTimer.

This avoids the macOS crash caused by calling cv2.imshow from
a background thread — all Qt/GUI work happens on the main thread.

Usage (in main_window.py):
    from ui.widgets.tracking_preview import TrackingPreviewWindow

    self.preview = TrackingPreviewWindow(self.movement_tracker)
    self.preview.show()

    # When done:
    self.preview.close()
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from typing import Optional


class TrackingPreviewWindow(QWidget):
    """
    Separate window showing the live hand tracking feed.

    Pulls frames from MovementTracker.get_preview_frame() at ~30 FPS
    using a QTimer (main-thread safe on macOS).
    """

    def __init__(self, tracker, parent=None):
        """
        Args:
            tracker: MovementTracker instance (must be started).
            parent: Parent QWidget.
        """
        super().__init__(parent)
        self.tracker = tracker

        self.setWindowTitle("Hand Tracking — EMS Movement Tracker")
        self.setMinimumSize(660, 520)
        self.resize(660, 520)

        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        # Timer to pull frames (~30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)

    def _update_frame(self) -> None:
        """Pull the latest annotated frame from the tracker and display it."""
        if self.tracker is None:
            return

        frame = self.tracker.get_preview_frame()
        if frame is None:
            return

        # Convert BGR (OpenCV) → RGB (Qt)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Scale to fit label
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def closeEvent(self, event) -> None:
        """Stop the timer when window is closed."""
        self.timer.stop()
        event.accept()

    def stop(self) -> None:
        """Stop updating (call before closing)."""
        self.timer.stop()
