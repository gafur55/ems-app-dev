"""
Session audio recorder.

Records microphone audio to a WAV file during an EMS session.
Designed for crash safety — audio is written directly to disk,
not buffered in memory.

Features:
    - Records to WAV file on disk (not in memory)
    - Auto-stops after MAX_DURATION_S (15 minutes)
    - Flushes to disk periodically (crash-safe)
    - Handles microphone unavailability gracefully

Requirements:
    pip install pyaudio

Usage:
    recorder = SessionRecorder(save_dir="data/recordings")
    filepath = recorder.start(session_id=1, participant_id="P001")
    ...
    recorder.stop()  # returns filepath
"""

import os
import time
import threading
import wave
import struct
from typing import Optional
from datetime import datetime

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("WARNING: pyaudio not installed — audio recording disabled")
    print("  Install with: pip install pyaudio")


class SessionRecorder:
    """
    Records session audio to a WAV file.

    Audio is written directly to disk in chunks, so even if the app
    crashes, most of the recording is preserved.
    """

    # Recording parameters
    SAMPLE_RATE = 44100
    CHANNELS = 1          # mono
    CHUNK_SIZE = 4096      # frames per buffer
    FORMAT_PA = None       # set in __init__ if pyaudio available
    SAMPLE_WIDTH = 2       # 16-bit = 2 bytes

    # Safety limits
    MAX_DURATION_S = 5 * 60  # 5 minutes

    def __init__(self, save_dir: str = "data/recordings"):
        """
        Args:
            save_dir: Directory to save WAV files.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self._recording = False
        self._thread: Optional[threading.Thread] = None
        self._filepath: Optional[str] = None
        self._wave_file: Optional[wave.Wave_write] = None
        self._start_time: float = 0.0
        self._pa: Optional[object] = None
        self._stream: Optional[object] = None
        self._lock = threading.Lock()
        self._frames_written = 0

        if PYAUDIO_AVAILABLE:
            self.FORMAT_PA = pyaudio.paInt16

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def filepath(self) -> Optional[str]:
        return self._filepath

    @property
    def elapsed_seconds(self) -> float:
        if self._recording:
            return time.time() - self._start_time
        return 0.0

    def start(self, session_id: int, participant_id: str = "") -> Optional[str]:
        """
        Start recording audio to a WAV file.

        Args:
            session_id: Database session ID (used in filename).
            participant_id: Participant ID (used in filename).

        Returns:
            Filepath of the recording, or None if recording failed to start.
        """
        if not PYAUDIO_AVAILABLE:
            print("[Recorder] pyaudio not available — skipping audio recording")
            return None

        if self._recording:
            print("[Recorder] Already recording")
            return self._filepath

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_{participant_id}_{timestamp}.wav"
        self._filepath = os.path.join(self.save_dir, filename)

        try:
            # Open PyAudio
            self._pa = pyaudio.PyAudio()

            # Find the default input device (laptop mic)
            device_info = self._pa.get_default_input_device_info()
            print(f"[Recorder] Using microphone: {device_info['name']}")

            # Open audio stream
            self._stream = self._pa.open(
                format=self.FORMAT_PA,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
            )

            # Open WAV file for writing
            self._wave_file = wave.open(self._filepath, 'wb')
            self._wave_file.setnchannels(self.CHANNELS)
            self._wave_file.setsampwidth(self.SAMPLE_WIDTH)
            self._wave_file.setframerate(self.SAMPLE_RATE)

            # Start recording thread
            self._recording = True
            self._start_time = time.time()
            self._frames_written = 0
            self._thread = threading.Thread(target=self._record_loop, daemon=True)
            self._thread.start()

            print(f"[Recorder] Recording started: {self._filepath}")
            print(f"[Recorder] Auto-stop in {self.MAX_DURATION_S // 60} minutes")
            return self._filepath

        except Exception as e:
            print(f"[Recorder] Failed to start recording: {e}")
            self._cleanup()
            return None

    def stop(self) -> Optional[str]:
        """
        Stop recording and finalize the WAV file.

        Returns:
            Filepath of the completed recording, or None.
        """
        if not self._recording:
            return self._filepath

        print("[Recorder] Stopping recording...")
        self._recording = False

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        # Finalize
        filepath = self._filepath
        self._cleanup()

        if filepath and os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            duration = self._frames_written / self.SAMPLE_RATE if self._frames_written > 0 else 0
            print(f"[Recorder] Recording saved: {filepath}")
            print(f"[Recorder]   Duration: {duration:.1f}s, Size: {size_mb:.1f} MB")
        else:
            print("[Recorder] No recording file saved")

        return filepath

    def _record_loop(self) -> None:
        """Background thread: reads mic data and writes to WAV file."""
        try:
            while self._recording:
                # Check max duration
                elapsed = time.time() - self._start_time
                if elapsed >= self.MAX_DURATION_S:
                    print(f"\n[Recorder] Max duration reached ({self.MAX_DURATION_S // 60} min) — auto-stopping")
                    self._recording = False
                    break

                # Read audio chunk
                try:
                    data = self._stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                except Exception as e:
                    print(f"[Recorder] Read error: {e}")
                    continue

                # Write to WAV file
                with self._lock:
                    if self._wave_file:
                        self._wave_file.writeframes(data)
                        self._frames_written += self.CHUNK_SIZE

        except Exception as e:
            print(f"[Recorder] Recording thread error: {e}")
        finally:
            # Ensure file is flushed even if thread exits unexpectedly
            self._flush_file()

    def _flush_file(self) -> None:
        """Flush WAV file to disk."""
        with self._lock:
            if self._wave_file:
                try:
                    self._wave_file.close()
                except Exception:
                    pass
                self._wave_file = None

    def _cleanup(self) -> None:
        """Release all audio resources."""
        # Close WAV file
        self._flush_file()

        # Close audio stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Terminate PyAudio
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None