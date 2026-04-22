"""
Session transcriber.

Transcribes WAV recordings using OpenAI Whisper (local, offline).
After successful transcription, deletes the WAV file to save disk space.

Runs in a background thread so it doesn't block the UI.

Requirements:
    pip install openai-whisper

Usage:
    transcriber = SessionTranscriber(db)
    transcriber.transcribe_recording(
        recording_id=1,
        session_id=1,
        participant_id="P001",
        wav_filepath="data/recordings/session_1_P001.wav",
        delete_after=True,
        callback=on_done,
    )
"""

import os
import time
import threading
from typing import Optional, Callable

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("WARNING: openai-whisper not installed — transcription disabled")
    print("  Install with: pip install openai-whisper")


# Whisper model size → VRAM / quality tradeoff
# "tiny"   — fastest, least accurate, ~1 GB RAM
# "base"   — good balance for conversational audio, ~1 GB RAM
# "small"  — better accuracy, ~2 GB RAM
# "medium" — high accuracy, ~5 GB RAM
# "large"  — best accuracy, ~10 GB RAM
DEFAULT_MODEL = "base"


class SessionTranscriber:
    """
    Transcribes session audio recordings using Whisper.

    - Loads Whisper model once, reuses across transcriptions
    - Runs transcription in background thread
    - Saves transcript to database
    - Optionally deletes WAV after successful transcription
    """

    def __init__(self, db, model_name: str = DEFAULT_MODEL):
        """
        Args:
            db: Database instance with transcription methods.
            model_name: Whisper model size.
        """
        self.db = db
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._busy = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    def _load_model(self):
        """Lazy-load Whisper model on first use."""
        if self._model is None:
            print(f"[Transcriber] Loading Whisper model '{self.model_name}'...")
            start = time.time()
            self._model = whisper.load_model(self.model_name)
            elapsed = time.time() - start
            print(f"[Transcriber] Model loaded in {elapsed:.1f}s")

    def transcribe_recording(
        self,
        recording_id: int,
        session_id: int,
        participant_id: str,
        wav_filepath: str,
        delete_after: bool = True,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Transcribe a WAV file in a background thread.

        Args:
            recording_id: DB recording ID.
            session_id: DB session ID.
            participant_id: Participant ID.
            wav_filepath: Path to WAV file.
            delete_after: Delete WAV after successful transcription.
            callback: Optional function called when done (success: bool, text: str).
        """
        if not WHISPER_AVAILABLE:
            print("[Transcriber] Whisper not available — skipping transcription")
            if callback:
                callback(False, "Whisper not installed")
            return

        if not os.path.exists(wav_filepath):
            print(f"[Transcriber] WAV file not found: {wav_filepath}")
            if callback:
                callback(False, "File not found")
            return

        thread = threading.Thread(
            target=self._transcribe_worker,
            args=(recording_id, session_id, participant_id,
                  wav_filepath, delete_after, callback),
            daemon=True,
        )
        thread.start()

    def _transcribe_worker(
        self,
        recording_id: int,
        session_id: int,
        participant_id: str,
        wav_filepath: str,
        delete_after: bool,
        callback: Optional[Callable],
    ) -> None:
        """Background worker: transcribe and save."""
        self._busy = True

        # Create pending record in DB
        transcription_id = self.db.create_transcription(
            recording_id=recording_id,
            session_id=session_id,
            participant_id=participant_id,
        )

        try:
            # Load model (first time only)
            self._load_model()

            # Transcribe
            print(f"[Transcriber] Transcribing: {wav_filepath}")
            start = time.time()

            result = self._model.transcribe(
                wav_filepath,
                language=None,  # auto-detect
                verbose=False,
            )

            elapsed = time.time() - start
            full_text = result.get("text", "").strip()
            language = result.get("language", "unknown")

            print(f"[Transcriber] Done in {elapsed:.1f}s")
            print(f"[Transcriber] Language: {language}")
            print(f"[Transcriber] Text length: {len(full_text)} chars")
            if full_text:
                preview = full_text[:200] + ("..." if len(full_text) > 200 else "")
                print(f"[Transcriber] Preview: {preview}")

            # Save to DB
            self.db.finalize_transcription(
                transcription_id=transcription_id,
                full_text=full_text,
                language=language,
                whisper_model=self.model_name,
                duration_s=elapsed,
            )

            # Delete WAV file after successful transcription
            if delete_after and full_text:
                try:
                    os.remove(wav_filepath)
                    self.db.mark_recording_audio_deleted(recording_id)
                    print(f"[Transcriber] WAV deleted: {wav_filepath}")
                except OSError as e:
                    print(f"[Transcriber] Failed to delete WAV: {e}")

            if callback:
                callback(True, full_text)

        except Exception as e:
            print(f"[Transcriber] Transcription failed: {e}")
            self.db.mark_transcription_failed(transcription_id, str(e))
            if callback:
                callback(False, str(e))

        finally:
            self._busy = False

    def transcribe_sync(self, wav_filepath: str) -> Optional[str]:
        """
        Synchronous transcription (for scripts/testing).

        Args:
            wav_filepath: Path to WAV file.

        Returns:
            Transcription text, or None on failure.
        """
        if not WHISPER_AVAILABLE:
            print("[Transcriber] Whisper not available")
            return None

        if not os.path.exists(wav_filepath):
            print(f"[Transcriber] File not found: {wav_filepath}")
            return None

        self._load_model()
        result = self._model.transcribe(wav_filepath, verbose=False)
        return result.get("text", "").strip()