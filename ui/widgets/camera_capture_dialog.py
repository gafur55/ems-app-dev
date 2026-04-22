import cv2
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import numpy as np


class CameraCaptureDialog(QDialog):
    """
    Dialog for capturing forearm image using webcam
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Forearm Image")
        self.setModal(True)
        self.resize(800, 600)
        
        self.captured_image = None  # Will store the captured frame
        self.camera = None
        self.timer = None
        
        self.setup_ui()
        self.start_camera()
    

    def setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: black;")
        layout.addWidget(self.video_label)
        
        # Instructions
        instructions = QLabel("Position your forearm in the frame and click Capture")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(instructions)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.capture_button = QPushButton("Capture")
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #00BF00;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00D000;
            }
        """)
        
        self.retake_button = QPushButton("Retake")
        self.retake_button.clicked.connect(self.retake)
        self.retake_button.setEnabled(False)
        self.retake_button.setStyleSheet("""
            QPushButton {
                background-color: #FFA500;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #FFB520;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        
        self.use_button = QPushButton("Use This Image")
        self.use_button.clicked.connect(self.accept)
        self.use_button.setEnabled(False)
        self.use_button.setStyleSheet("""
            QPushButton {
                background-color: #0080FF;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0090FF;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet("padding: 10px 20px; font-size: 14px;")
        
        button_layout.addStretch()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.retake_button)
        button_layout.addWidget(self.use_button)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)


    def start_camera(self):
        """Initialize and start the camera feed"""
        self.camera = cv2.VideoCapture(0)  # 0 = default camera
        
        if not self.camera.isOpened():
            self.video_label.setText("Error: Could not open camera")
            self.capture_button.setEnabled(False)
            return
        
        # Set camera resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (~33 FPS)
    

    def update_frame(self):
        """Update the video feed with current camera frame"""
        ret, frame = self.camera.read()
        
        if ret:
            # Convert BGR (OpenCV) to RGB (Qt)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Display in label
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
    

    def capture_frame(self):
        """Capture current frame and freeze video"""
        ret, frame = self.camera.read()
        
        if ret:
            # Store the captured frame
            self.captured_image = frame.copy()
            
            # Stop video feed
            if self.timer:
                self.timer.stop()
            
            # Display captured frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            
            # Update button states
            self.capture_button.setEnabled(False)
            self.retake_button.setEnabled(True)
            self.use_button.setEnabled(True)

            print("Frame captured!")
    

    def retake(self):
        """Restart video feed for retaking photo"""
        # Restart timer
        if self.timer:
            self.timer.start(30)
        
        # Update button states
        self.capture_button.setEnabled(True)
        self.retake_button.setEnabled(False)
        self.use_button.setEnabled(False)

        print("Ready to capture again")
    

    def get_captured_image(self):
        """Return the captured image and its dimensions"""
        if self.captured_image is not None:
            height, width = self.captured_image.shape[:2]
            return self.captured_image, width, height
        return None, None, None
    

    def closeEvent(self, event):
        """Clean up camera when dialog closes"""
        if self.timer:
            self.timer.stop()
        if self.camera:
            self.camera.release()
        event.accept()