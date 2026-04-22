"""
EMS Experiment Database — SQLite storage for stimulation & movement data.

Designed for:
- Local-first: single .db file, no server, travels with the code
- MLOps-ready: flat export to CSV/pandas DataFrames for ML pipelines
- Multi-researcher: each researcher runs locally, merge DBs later
- Reproducibility: stores everything needed to reproduce results

Schema overview:
    participants     → who (age, arm measurements, experience)
    sessions         → when + setup (device, calibration, target gesture)
    electrodes       → where (anatomical coordinates per session)
    stimulations     → what (channel, intensity, pulse width, pulse count)
    movement_results → outcome (wrist angle, finger angles, peak timing)
    joint_angles     → detail (per-joint per-finger angles for each stimulation)

Usage:
    from data.database import EMSDatabase

    db = EMSDatabase()                          # creates ems_data.db
    db = EMSDatabase("experiment_2024.db")      # custom name

    # Store a session
    session_id = db.create_session(participant_id, device_name="p24", ...)

    # Store stimulation + movement result
    stim_id = db.record_stimulation(session_id, channel=0, intensity=15, ...)
    db.record_movement(stim_id, movement_result)

    # Export for ML
    df = db.export_flat_dataframe()             # everything in one flat table
    db.export_csv("training_data.csv")          # or straight to CSV
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


# Default database location: data/ folder next to the app
DEFAULT_DB_DIR = "/Users/gafurmammadov/Documents/Uchicago_classes/practicum/project_database"
DEFAULT_DB_NAME = "ems_data.db"


class EMSDatabase:
    """
    SQLite database for EMS experiment data.

    The database file is created automatically on first use.
    All tables are created if they don't exist.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Full path to .db file. If None, uses data/ems_data.db
        """
        if db_path is None:
            os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
            db_path = os.path.join(DEFAULT_DB_DIR, DEFAULT_DB_NAME)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # dict-like access
        self.conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        self.conn.execute("PRAGMA foreign_keys=ON")

        self._create_tables()
        self._migrate()
        print(f"✓ Database ready: {os.path.abspath(db_path)}")

    # ==================================================================
    # Schema
    # ==================================================================

    def _migrate(self) -> None:
        """Add columns that may not exist in older databases."""
        # Add name column to participants
        try:
            self.conn.execute("SELECT name FROM participants LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE participants ADD COLUMN name TEXT")
            self.conn.commit()
            print("  DB: Migrated — added 'name' column to participants")

        # Add stim_id column to electrodes (links positions to each stimulation)
        try:
            self.conn.execute("SELECT stim_id FROM electrodes LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE electrodes ADD COLUMN stim_id INTEGER REFERENCES stimulations(stim_id)")
            self.conn.commit()
            print("  DB: Migrated — added 'stim_id' column to electrodes")

    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        c = self.conn.cursor()

        # --- Participants ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                participant_id  TEXT PRIMARY KEY,
                name            TEXT,
                age             INTEGER,
                arm_width_cm    REAL,
                arm_length_cm   REAL,
                skin_resistance_100hz_kohm  REAL,
                skin_resistance_1khz_kohm   REAL,
                skin_resistance_10khz_kohm  REAL,
                skin_resistance_100khz_kohm REAL,
                pain_threshold_ma REAL,
                experience_level TEXT,
                notes           TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)

        # --- Sessions ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id  TEXT REFERENCES participants(participant_id),
                device_name     TEXT NOT NULL,
                target_gesture  TEXT,
                calibration_factor REAL,
                forearm_length_cm  REAL,
                origin_pixel_x  REAL,
                origin_pixel_y  REAL,
                notes           TEXT,
                started_at      TEXT DEFAULT (datetime('now')),
                ended_at        TEXT
            )
        """)

        # --- Session Recordings ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS session_recordings (
                recording_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER REFERENCES sessions(session_id),
                participant_id  TEXT REFERENCES participants(participant_id),
                filepath        TEXT NOT NULL,
                duration_s      REAL,
                file_size_bytes INTEGER,
                started_at      TEXT DEFAULT (datetime('now')),
                ended_at        TEXT,
                status          TEXT DEFAULT 'recording'
            )
        """)

        # --- Session Transcriptions ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS session_transcriptions (
                transcription_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id     INTEGER REFERENCES session_recordings(recording_id),
                session_id       INTEGER REFERENCES sessions(session_id),
                participant_id   TEXT REFERENCES participants(participant_id),
                full_text        TEXT,
                language         TEXT,
                whisper_model    TEXT,
                duration_s       REAL,
                transcribed_at   TEXT DEFAULT (datetime('now')),
                status           TEXT DEFAULT 'pending'
            )
        """)

        # --- Electrodes (per stimulation) ---
        # Saved with every stimulation so electrode moves are tracked.
        # Coordinate system (after calibration):
        #   real_x_cm:  displacement from wrist midline
        #                (+) = radial (thumb side), (-) = ulnar (pinky side)
        #   real_y_cm: distance along forearm from wrist toward elbow
        #                0 = at wrist, forearm_length = at elbow
        c.execute("""
            CREATE TABLE IF NOT EXISTS electrodes (
                electrode_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                stim_id         INTEGER REFERENCES stimulations(stim_id),
                session_id      INTEGER NOT NULL REFERENCES sessions(session_id),
                channel         INTEGER NOT NULL,
                pixel_x         REAL,
                pixel_y         REAL,
                real_x_cm      REAL,
                real_y_cm     REAL,
                normalized_s    REAL,
                normalized_t    REAL,
                distance_from_wrist_cm REAL,
                anatomical_zone TEXT,
                placement_method TEXT DEFAULT 'manual',
                placed_at       TEXT DEFAULT (datetime('now'))
            )
        """)

        # --- Stimulations ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS stimulations (
                stim_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL REFERENCES sessions(session_id),
                channel         INTEGER NOT NULL,
                intensity_ma    INTEGER NOT NULL,
                pulse_width_us  INTEGER NOT NULL,
                pulse_count     INTEGER,
                delay_ms        REAL,
                stim_duration_ms REAL,
                stimulated_at   TEXT DEFAULT (datetime('now'))
            )
        """)

        # --- Movement Results (one per stimulation) ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS movement_results (
                result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                stim_id         INTEGER NOT NULL UNIQUE REFERENCES stimulations(stim_id),
                
                -- Wrist
                wrist_angle_delta    REAL,
                wrist_direction      TEXT,
                
                -- Finger summary
                total_finger_movement REAL,
                primary_finger       TEXT,
                primary_finger_movement REAL,
                movement_type        TEXT,
                
                -- Per-finger flexion change (total per finger)
                thumb_flexion_change  REAL,
                index_flexion_change  REAL,
                middle_flexion_change REAL,
                ring_flexion_change   REAL,
                pinky_flexion_change  REAL,
                
                -- Quality
                peak_frame_index     INTEGER,
                total_frames         INTEGER,
                peak_latency_ms      REAL,
                confidence           REAL,
                hand_detected        INTEGER,
                
                recorded_at     TEXT DEFAULT (datetime('now'))
            )
        """)

        # --- Joint Angles (per-joint detail, 15 rows per stimulation) ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS joint_angles (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                stim_id         INTEGER NOT NULL REFERENCES stimulations(stim_id),
                finger          TEXT NOT NULL,
                joint           TEXT NOT NULL,
                baseline_angle  REAL,
                result_angle    REAL,
                angle_delta     REAL,
                
                UNIQUE(stim_id, finger, joint)
            )
        """)

        # --- Indexes for common queries ---
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_stim_session 
            ON stimulations(session_id)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_stim 
            ON movement_results(stim_id)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_joints_stim 
            ON joint_angles(stim_id)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_electrodes_session 
            ON electrodes(session_id)
        """)

        self.conn.commit()

    # ==================================================================
    # Write operations
    # ==================================================================

    def create_participant(
        self,
        participant_id: str,
        name: Optional[str] = None,
        age: Optional[int] = None,
        arm_width_cm: Optional[float] = None,
        arm_length_cm: Optional[float] = None,
        skin_resistance_100hz_kohm: Optional[float] = None,
        skin_resistance_1khz_kohm: Optional[float] = None,
        skin_resistance_10khz_kohm: Optional[float] = None,
        skin_resistance_100khz_kohm: Optional[float] = None,
        pain_threshold_ma: Optional[float] = None,
        experience_level: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Create or update a participant.

        Returns:
            participant_id
        """
        self.conn.execute("""
            INSERT INTO participants 
                (participant_id, name, age, arm_width_cm, arm_length_cm, 
                 skin_resistance_100hz_kohm, skin_resistance_1khz_kohm,
                 skin_resistance_10khz_kohm, skin_resistance_100khz_kohm,
                 pain_threshold_ma, experience_level, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id) DO UPDATE SET
                name=excluded.name,
                age=excluded.age,
                arm_width_cm=excluded.arm_width_cm,
                arm_length_cm=excluded.arm_length_cm,
                skin_resistance_100hz_kohm=excluded.skin_resistance_100hz_kohm,
                skin_resistance_1khz_kohm=excluded.skin_resistance_1khz_kohm,
                skin_resistance_10khz_kohm=excluded.skin_resistance_10khz_kohm,
                skin_resistance_100khz_kohm=excluded.skin_resistance_100khz_kohm,
                pain_threshold_ma=excluded.pain_threshold_ma,
                experience_level=excluded.experience_level,
                notes=excluded.notes
        """, (participant_id, name, age, arm_width_cm, arm_length_cm,
              skin_resistance_100hz_kohm, skin_resistance_1khz_kohm,
              skin_resistance_10khz_kohm, skin_resistance_100khz_kohm,
              pain_threshold_ma, experience_level, notes))
        self.conn.commit()
        return participant_id

    def create_session(
        self,
        participant_id: str,
        device_name: str = "p24",
        target_gesture: Optional[str] = None,
        calibration_factor: Optional[float] = None,
        forearm_length_cm: Optional[float] = None,
        origin_pixel_x: Optional[float] = None,
        origin_pixel_y: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Create a new session.

        Returns:
            session_id
        """
        cursor = self.conn.execute("""
            INSERT INTO sessions 
                (participant_id, device_name, target_gesture,
                 calibration_factor, forearm_length_cm,
                 origin_pixel_x, origin_pixel_y, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (participant_id, device_name, target_gesture,
              calibration_factor, forearm_length_cm,
              origin_pixel_x, origin_pixel_y, notes))
        self.conn.commit()
        session_id = cursor.lastrowid
        print(f"  DB: Session {session_id} created for {participant_id}")
        return session_id

    def end_session(self, session_id: int) -> None:
        """Mark session as ended."""
        self.conn.execute("""
            UPDATE sessions SET ended_at = datetime('now') WHERE session_id = ?
        """, (session_id,))
        self.conn.commit()

    def get_session_max_intensity(self, session_id: int) -> Optional[float]:
        """
        Get the maximum intensity_ma used in a session.

        Returns:
            Max intensity in mA, or None if no stimulations recorded.
        """
        cursor = self.conn.execute("""
            SELECT MAX(intensity_ma) FROM stimulations WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None

    def update_pain_threshold_ma(self, participant_id: str, pain_threshold_ma: float) -> None:
        """
        Update the participant's pain threshold (max tolerated intensity in mA).
        Only updates if the new value is higher than the existing one.

        Args:
            participant_id: Participant ID
            pain_threshold_ma: Max tolerated intensity in mA
        """
        self.conn.execute("""
            UPDATE participants 
            SET pain_threshold_ma = MAX(COALESCE(pain_threshold_ma, 0), ?)
            WHERE participant_id = ?
        """, (pain_threshold_ma, participant_id))
        self.conn.commit()
        print(f"  DB: Pain threshold for {participant_id} updated to {pain_threshold_ma} mA")

    # ------------------------------------------------------------------
    # Session recordings
    # ------------------------------------------------------------------

    def start_recording(
        self,
        session_id: int,
        participant_id: str,
        filepath: str,
    ) -> int:
        """
        Record that a session audio recording has started.
        Status is 'recording' until finalized.

        Returns:
            recording_id
        """
        cursor = self.conn.execute("""
            INSERT INTO session_recordings
                (session_id, participant_id, filepath, status)
            VALUES (?, ?, ?, 'recording')
        """, (session_id, participant_id, filepath))
        self.conn.commit()
        recording_id = cursor.lastrowid
        print(f"  DB: Recording {recording_id} started for session {session_id}")
        return recording_id

    def finalize_recording(
        self,
        recording_id: int,
        duration_s: float,
        file_size_bytes: int,
    ) -> None:
        """
        Mark a recording as completed with final metadata.
        """
        self.conn.execute("""
            UPDATE session_recordings
            SET ended_at = datetime('now'),
                duration_s = ?,
                file_size_bytes = ?,
                status = 'completed'
            WHERE recording_id = ?
        """, (duration_s, file_size_bytes, recording_id))
        self.conn.commit()
        print(f"  DB: Recording {recording_id} finalized ({duration_s:.1f}s)")

    def mark_recording_interrupted(self, recording_id: int) -> None:
        """
        Mark a recording as interrupted (crash/unexpected close).
        The WAV file may still be partially usable.
        """
        self.conn.execute("""
            UPDATE session_recordings
            SET ended_at = datetime('now'),
                status = 'interrupted'
            WHERE recording_id = ? AND status = 'recording'
        """, (recording_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Session transcriptions
    # ------------------------------------------------------------------

    def create_transcription(
        self,
        recording_id: int,
        session_id: int,
        participant_id: str,
    ) -> int:
        """
        Create a pending transcription record.

        Returns:
            transcription_id
        """
        cursor = self.conn.execute("""
            INSERT INTO session_transcriptions
                (recording_id, session_id, participant_id, status)
            VALUES (?, ?, ?, 'pending')
        """, (recording_id, session_id, participant_id))
        self.conn.commit()
        return cursor.lastrowid

    def finalize_transcription(
        self,
        transcription_id: int,
        full_text: str,
        language: str,
        whisper_model: str,
        duration_s: float,
    ) -> None:
        """
        Store completed transcription text and metadata.
        """
        self.conn.execute("""
            UPDATE session_transcriptions
            SET full_text = ?,
                language = ?,
                whisper_model = ?,
                duration_s = ?,
                transcribed_at = datetime('now'),
                status = 'completed'
            WHERE transcription_id = ?
        """, (full_text, language, whisper_model, duration_s, transcription_id))
        self.conn.commit()
        print(f"  DB: Transcription {transcription_id} saved ({len(full_text)} chars)")

    def mark_transcription_failed(self, transcription_id: int, error: str) -> None:
        """Mark a transcription as failed."""
        self.conn.execute("""
            UPDATE session_transcriptions
            SET status = 'failed',
                full_text = ?
            WHERE transcription_id = ?
        """, (f"ERROR: {error}", transcription_id))
        self.conn.commit()

    def get_recording_filepath(self, recording_id: int) -> Optional[str]:
        """Get the WAV filepath for a recording."""
        cursor = self.conn.execute("""
            SELECT filepath FROM session_recordings WHERE recording_id = ?
        """, (recording_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def mark_recording_audio_deleted(self, recording_id: int) -> None:
        """Mark that the WAV file has been deleted after transcription."""
        self.conn.execute("""
            UPDATE session_recordings
            SET status = 'transcribed_and_deleted'
            WHERE recording_id = ?
        """, (recording_id,))
        self.conn.commit()

    def record_electrode(
        self,
        session_id: int,
        channel: int,
        pixel_x: float,
        pixel_y: float,
        stim_id: Optional[int] = None,
        real_x_cm: Optional[float] = None,
        real_y_cm: Optional[float] = None,
        normalized_s: Optional[float] = None,
        normalized_t: Optional[float] = None,
        distance_from_wrist_cm: Optional[float] = None,
        anatomical_zone: Optional[str] = None,
        placement_method: str = "manual",
    ) -> int:
        """
        Record an electrode placement.

        Args:
            session_id: Session this electrode belongs to.
            channel: EMS channel number.
            pixel_x: X position in image pixels.
            pixel_y: Y position in image pixels.
            stim_id: Stimulation this snapshot belongs to (None for initial placement).
            real_x_cm: Lateral displacement from wrist midline (cm).
                        (+) = radial/thumb side, (-) = ulnar/pinky side.
            real_y_cm: Distance along forearm from wrist toward elbow (cm).
                         0 = at wrist, forearm_length = at elbow.

        Returns:
            electrode_id
        """
        cursor = self.conn.execute("""
            INSERT INTO electrodes
                (stim_id, session_id, channel, pixel_x, pixel_y,
                 real_x_cm, real_y_cm,
                 normalized_s, normalized_t, distance_from_wrist_cm,
                 anatomical_zone, placement_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (stim_id, session_id, channel, pixel_x, pixel_y,
              real_x_cm, real_y_cm,
              normalized_s, normalized_t, distance_from_wrist_cm,
              anatomical_zone, placement_method))
        self.conn.commit()
        return cursor.lastrowid

    def record_stimulation(
        self,
        session_id: int,
        channel: int,
        intensity_ma: int,
        pulse_width_us: int,
        pulse_count: Optional[int] = None,
        delay_ms: Optional[float] = None,
    ) -> int:
        """
        Record a stimulation event.

        Returns:
            stim_id
        """
        stim_duration_ms = None
        if pulse_count and delay_ms:
            stim_duration_ms = pulse_count * delay_ms

        cursor = self.conn.execute("""
            INSERT INTO stimulations
                (session_id, channel, intensity_ma, pulse_width_us,
                 pulse_count, delay_ms, stim_duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, channel, intensity_ma, pulse_width_us,
              pulse_count, delay_ms, stim_duration_ms))
        self.conn.commit()
        stim_id = cursor.lastrowid
        print(f"  DB: Stimulation {stim_id} recorded (Ch{channel}, {intensity_ma}mA)")
        return stim_id

    def record_movement(
        self,
        stim_id: int,
        result,  # StimulationMovementResult
        peak_frame_index: Optional[int] = None,
        total_frames: Optional[int] = None,
    ) -> int:
        """
        Record movement result for a stimulation.

        Args:
            stim_id: The stimulation this result belongs to.
            result: StimulationMovementResult from MovementTracker.
            peak_frame_index: Which frame was the peak.
            total_frames: How many frames were recorded.

        Returns:
            result_id
        """
        # Insert main result row
        cursor = self.conn.execute("""
            INSERT INTO movement_results
                (stim_id, wrist_angle_delta, wrist_direction,
                 total_finger_movement, primary_finger, primary_finger_movement,
                 movement_type,
                 thumb_flexion_change, index_flexion_change,
                 middle_flexion_change, ring_flexion_change, pinky_flexion_change,
                 peak_frame_index, total_frames, peak_latency_ms,
                 confidence, hand_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stim_id,
            result.wrist_angle_delta,
            result.wrist_direction,
            result.total_finger_movement,
            result.primary_finger,
            result.primary_finger_movement,
            result.movement_type,
            result.flexion_change.get("thumb", 0),
            result.flexion_change.get("index", 0),
            result.flexion_change.get("middle", 0),
            result.flexion_change.get("ring", 0),
            result.flexion_change.get("pinky", 0),
            peak_frame_index,
            total_frames,
            result.latency_ms,
            result.confidence,
            1 if result.hand_detected else 0,
        ))

        result_id = cursor.lastrowid

        # Insert per-joint angle details (15 rows: 5 fingers × 3 joints)
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        joint_names = {
            "thumb": ["CMC", "MCP", "IP"],
            "index": ["MCP", "PIP", "DIP"],
            "middle": ["MCP", "PIP", "DIP"],
            "ring": ["MCP", "PIP", "DIP"],
            "pinky": ["MCP", "PIP", "DIP"],
        }

        for finger in finger_names:
            joints = joint_names[finger]
            baseline = result.baseline_finger_angles.get(finger, [0, 0, 0])
            result_angles = result.result_finger_angles.get(finger, [0, 0, 0])
            deltas = result.finger_angle_deltas.get(finger, [0, 0, 0])

            for i, joint in enumerate(joints):
                self.conn.execute("""
                    INSERT INTO joint_angles
                        (stim_id, finger, joint, baseline_angle, result_angle, angle_delta)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    stim_id, finger, joint,
                    baseline[i] if i < len(baseline) else 0,
                    result_angles[i] if i < len(result_angles) else 0,
                    deltas[i] if i < len(deltas) else 0,
                ))

        self.conn.commit()
        print(f"  DB: Movement result {result_id} recorded for stim {stim_id}")
        return result_id

    # ==================================================================
    # Read / Query operations
    # ==================================================================

    def get_all_participants(self) -> List[Dict]:
        """Get all participants ordered by most recent."""
        rows = self.conn.execute("""
            SELECT p.*, 
                   COUNT(DISTINCT s.session_id) as session_count,
                   MAX(s.started_at) as last_session
            FROM participants p
            LEFT JOIN sessions s ON p.participant_id = s.participant_id
            GROUP BY p.participant_id
            ORDER BY last_session DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_participant(self, participant_id: str) -> Optional[Dict]:
        """Get a single participant by ID."""
        row = self.conn.execute("""
            SELECT * FROM participants WHERE participant_id = ?
        """, (participant_id,)).fetchone()
        return dict(row) if row else None

    def get_session_summary(self, session_id: int) -> Dict:
        """Get summary of a session."""
        row = self.conn.execute("""
            SELECT s.*, p.age, p.experience_level,
                   COUNT(DISTINCT st.stim_id) as stim_count
            FROM sessions s
            LEFT JOIN participants p ON s.participant_id = p.participant_id
            LEFT JOIN stimulations st ON s.session_id = st.session_id
            WHERE s.session_id = ?
            GROUP BY s.session_id
        """, (session_id,)).fetchone()

        return dict(row) if row else {}

    def get_session_results(self, session_id: int) -> List[Dict]:
        """Get all stimulation results for a session."""
        rows = self.conn.execute("""
            SELECT st.*, mr.*
            FROM stimulations st
            LEFT JOIN movement_results mr ON st.stim_id = mr.stim_id
            WHERE st.session_id = ?
            ORDER BY st.stimulated_at
        """, (session_id,)).fetchall()

        return [dict(r) for r in rows]

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions with participant info."""
        rows = self.conn.execute("""
            SELECT s.session_id, s.participant_id, s.device_name,
                   s.started_at, s.ended_at,
                   COUNT(DISTINCT st.stim_id) as stim_count,
                   p.age, p.experience_level
            FROM sessions s
            LEFT JOIN participants p ON s.participant_id = p.participant_id
            LEFT JOIN stimulations st ON s.session_id = st.session_id
            GROUP BY s.session_id
            ORDER BY s.started_at DESC
        """).fetchall()

        return [dict(r) for r in rows]

    # ==================================================================
    # ML Export
    # ==================================================================

    def export_flat_dataframe(self):
        """
        Export ALL data as a single flat pandas DataFrame.

        One row per stimulation with all context columns.
        Ready for ML training.

        Returns:
            pandas.DataFrame
        """
        import pandas as pd

        query = """
            SELECT
                -- Session context
                s.session_id,
                s.participant_id,
                s.device_name,
                s.target_gesture,
                s.forearm_length_cm,
                s.started_at as session_started,
                
                -- Participant
                p.name,
                p.age,
                p.arm_width_cm,
                p.arm_length_cm,
                p.skin_resistance_100hz_kohm,
                p.skin_resistance_1khz_kohm,
                p.skin_resistance_10khz_kohm,
                p.skin_resistance_100khz_kohm,
                p.pain_threshold_ma,
                p.experience_level,
                
                -- Stimulation parameters
                st.stim_id,
                st.channel,
                st.intensity_ma,
                st.pulse_width_us,
                st.pulse_count,
                st.delay_ms,
                st.stim_duration_ms,
                st.stimulated_at,
                
                -- Movement results
                mr.wrist_angle_delta,
                mr.wrist_direction,
                mr.total_finger_movement,
                mr.primary_finger,
                mr.primary_finger_movement,
                mr.movement_type,
                mr.thumb_flexion_change,
                mr.index_flexion_change,
                mr.middle_flexion_change,
                mr.ring_flexion_change,
                mr.pinky_flexion_change,
                mr.peak_frame_index,
                mr.total_frames,
                mr.peak_latency_ms,
                mr.confidence,
                mr.hand_detected,

                -- Session recording
                sr.recording_id,
                sr.duration_s as recording_duration_s,
                sr.file_size_bytes as recording_file_size_bytes,
                sr.status as recording_status,

                -- Session transcription
                stx.transcription_id,
                stx.full_text as transcription_text,
                stx.language as transcription_language,
                stx.whisper_model,
                stx.duration_s as transcription_duration_s,
                stx.status as transcription_status

            FROM stimulations st
            JOIN sessions s ON st.session_id = s.session_id
            LEFT JOIN participants p ON s.participant_id = p.participant_id
            LEFT JOIN movement_results mr ON st.stim_id = mr.stim_id
            LEFT JOIN session_recordings sr ON s.session_id = sr.session_id
            LEFT JOIN session_transcriptions stx ON s.session_id = stx.session_id
            ORDER BY st.stimulated_at
        """

        df = pd.read_sql_query(query, self.conn)

        # Also add per-joint columns
        joints_df = self._export_joint_angles_wide()
        if joints_df is not None and not joints_df.empty:
            df = df.merge(joints_df, on="stim_id", how="left")

        # Add electrode positions per stimulation
        electrodes_df = self._export_electrodes_wide()
        if electrodes_df is not None and not electrodes_df.empty:
            df = df.merge(electrodes_df, on="stim_id", how="left")

        return df

    def _export_joint_angles_wide(self):
        """Pivot joint_angles table into wide format (one column per joint)."""
        import pandas as pd

        rows = self.conn.execute("""
            SELECT stim_id, finger, joint, 
                   baseline_angle, result_angle, angle_delta
            FROM joint_angles
        """).fetchall()

        if not rows:
            return None

        records = [dict(r) for r in rows]
        df = pd.DataFrame(records)

        # Create column names like: thumb_CMC_baseline, thumb_CMC_result, thumb_CMC_delta
        wide_parts = []
        for metric, col in [("baseline", "baseline_angle"),
                            ("result", "result_angle"),
                            ("delta", "angle_delta")]:
            pivoted = df.pivot_table(
                index="stim_id",
                columns=["finger", "joint"],
                values=col,
                aggfunc="first"
            )
            pivoted.columns = [f"{f}_{j}_{metric}" for f, j in pivoted.columns]
            wide_parts.append(pivoted)

        wide = pd.concat(wide_parts, axis=1).reset_index()
        return wide

    def _export_electrodes_wide(self):
        """
        Pivot electrode positions into wide format per stimulation.

        Each stimulation has electrode positions saved with it.
        Electrodes become columns like:
            electrode_1_real_x_cm, electrode_1_real_y_cm, ...

        Coordinate system:
            real_x_cm:  displacement from wrist midline
                         (+) = radial/thumb side, (-) = ulnar/pinky side
            real_y_cm: distance along forearm from wrist toward elbow
                         0 = at wrist, forearm_length = at elbow
        """
        import pandas as pd

        rows = self.conn.execute("""
            SELECT stim_id, channel, pixel_x, pixel_y,
                   real_x_cm, real_y_cm,
                   normalized_s, normalized_t,
                   distance_from_wrist_cm, anatomical_zone
            FROM electrodes
            WHERE stim_id IS NOT NULL
            ORDER BY stim_id, channel, electrode_id
        """).fetchall()

        if not rows:
            return None

        records = [dict(r) for r in rows]
        df = pd.DataFrame(records)

        # Number electrodes within each stimulation
        df["electrode_num"] = df.groupby("stim_id").cumcount() + 1

        # Pivot each measurement into its own column
        result_parts = []
        for col in ["pixel_x", "pixel_y", "real_x_cm", "real_y_cm",
                     "normalized_s", "normalized_t",
                     "distance_from_wrist_cm", "anatomical_zone"]:
            pivoted = df.pivot_table(
                index="stim_id",
                columns="electrode_num",
                values=col,
                aggfunc="first"
            )
            pivoted.columns = [f"electrode_{int(n)}_{col}" for n in pivoted.columns]
            result_parts.append(pivoted)

        if not result_parts:
            return None

        wide = pd.concat(result_parts, axis=1).reset_index()
        return wide

    def export_csv(self, filepath: str = "ems_training_data.csv") -> str:
        """
        Export flat data to CSV file.

        Args:
            filepath: Output CSV path.

        Returns:
            Absolute path to the CSV file.
        """
        df = self.export_flat_dataframe()
        df.to_csv(filepath, index=False)
        abs_path = os.path.abspath(filepath)
        print(f"✓ Exported {len(df)} rows to {abs_path}")
        return abs_path

    def export_electrode_csv(self, filepath: str = "electrode_placements.csv") -> str:
        """Export electrode placement data to CSV."""
        import pandas as pd

        query = """
            SELECT e.*, s.participant_id, s.device_name
            FROM electrodes e
            JOIN sessions s ON e.session_id = s.session_id
            ORDER BY e.session_id, e.channel
        """
        df = pd.read_sql_query(query, self.conn)
        df.to_csv(filepath, index=False)
        print(f"✓ Exported {len(df)} electrode placements to {os.path.abspath(filepath)}")
        return os.path.abspath(filepath)

    # ==================================================================
    # Stats
    # ==================================================================

    def print_stats(self) -> None:
        """Print database statistics."""
        counts = {}
        for table in ["participants", "sessions", "stimulations",
                       "movement_results", "joint_angles", "electrodes",
                       "session_recordings", "session_transcriptions"]:
            try:
                row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0]
            except Exception:
                counts[table] = 0

        print()
        print("=" * 40)
        print("  EMS Database Stats")
        print("=" * 40)
        print(f"  Participants:      {counts['participants']}")
        print(f"  Sessions:          {counts['sessions']}")
        print(f"  Electrode places:  {counts['electrodes']}")
        print(f"  Stimulations:      {counts['stimulations']}")
        print(f"  Movement results:  {counts['movement_results']}")
        print(f"  Joint angle rows:  {counts['joint_angles']}")
        print(f"  Recordings:        {counts['session_recordings']}")
        print(f"  Transcriptions:    {counts['session_transcriptions']}")
        print(f"  Database file:     {os.path.abspath(self.db_path)}")
        print("=" * 40)
        print()

    # ==================================================================
    # Cleanup
    # ==================================================================

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        print(f"  DB: Connection closed")


# ==================================================================
# Quick test
# ==================================================================

if __name__ == "__main__":
    """Quick self-test — creates a test DB and verifies it works."""
    import tempfile

    test_path = os.path.join(tempfile.gettempdir(), "ems_test.db")
    db = EMSDatabase(test_path)

    # Create participant
    db.create_participant("P001", age=25, arm_length_cm=28.0, experience_level="novice")

    # Create session
    sid = db.create_session("P001", device_name="p24", forearm_length_cm=28.0)

    # Record electrode
    db.record_electrode(sid, channel=0, pixel_x=100, pixel_y=200,
                        real_x_cm=2.5, real_y_cm=5.0, normalized_s=0.3, normalized_t=0.1)

    # Record stimulation
    stim_id = db.record_stimulation(sid, channel=0, intensity_ma=15,
                                     pulse_width_us=250, pulse_count=10, delay_ms=10)

    # Simulate a movement result
    from dataclasses import dataclass, field
    from typing import Dict, List

    @dataclass
    class FakeResult:
        wrist_angle_delta: float = -24.2
        wrist_direction: str = "downward"
        total_finger_movement: float = 98.8
        primary_finger: str = "index"
        primary_finger_movement: float = -38.7
        movement_type: str = "flexion"
        flexion_change: Dict[str, float] = field(default_factory=lambda: {
            "thumb": 3.9, "index": -38.7, "middle": -32.7, "ring": -16.7, "pinky": -6.8
        })
        baseline_finger_angles: Dict[str, List[float]] = field(default_factory=lambda: {
            "thumb": [150, 160, 170], "index": [160, 155, 165],
            "middle": [158, 152, 163], "ring": [155, 150, 160], "pinky": [150, 148, 158]
        })
        result_finger_angles: Dict[str, List[float]] = field(default_factory=lambda: {
            "thumb": [152, 161, 172], "index": [148, 137, 158],
            "middle": [148, 140, 156], "ring": [149, 143, 157], "pinky": [148, 146, 155]
        })
        finger_angle_deltas: Dict[str, List[float]] = field(default_factory=lambda: {
            "thumb": [2, 1, 2], "index": [-12, -18, -7],
            "middle": [-10, -12, -7], "ring": [-6, -7, -3], "pinky": [-2, -2, -3]
        })
        latency_ms: float = 680
        confidence: float = 0.96
        hand_detected: bool = True

    result = FakeResult()
    db.record_movement(stim_id, result, peak_frame_index=34, total_frames=78)

    db.end_session(sid)
    db.print_stats()

    # Test export
    try:
        import pandas as pd
        df = db.export_flat_dataframe()
        print(f"Flat DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
    except ImportError:
        print("pandas not installed — skipping DataFrame test")

    db.close()

    # Cleanup
    os.remove(test_path)
    print(f"\n✓ All tests passed — test DB cleaned up")