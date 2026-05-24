# EMS Control GUI — System Documentation

A research application for running **Electrical Muscle Stimulation (EMS)** experiments on the forearm. The system stimulates muscles via an EMS device, simultaneously tracks the resulting wrist and finger movement using three cameras, and records every detail to a SQLite database for later analysis and ML training.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Hardware Setup](#2-hardware-setup)
3. [Architecture Overview](#3-architecture-overview)
4. [The Full Workflow (Step-by-Step)](#4-the-full-workflow-step-by-step)
5. [Component Deep Dive](#5-component-deep-dive)
6. [Data Storage](#6-data-storage)
7. [Coordinate Systems Explained](#7-coordinate-systems-explained)
8. [File Map](#8-file-map)
9. [Running the App](#9-running-the-app)
10. [Hotkeys & Tips](#10-hotkeys--tips)

---

## 1. What This System Does

In one sentence: **press a button, an EMS device shocks the participant's forearm, three cameras measure what their hand does, and everything gets saved to a database.**

The goal is to build a dataset that links **electrode position + stimulation parameters → resulting movement**, so that later you can predict (or even invert) the relationship: "I want the index finger to flex 30° — where should I put the electrodes and how much current?"

### Each experiment captures:
- **Who** — participant ID, age, arm dimensions, skin resistance at 4 frequencies, experience level, pain threshold.
- **Where** — exact electrode positions (in mm) relative to the wrist, derived automatically from ArUco markers.
- **What** — channel, intensity (mA), pulse width (μs), pulse count, delay.
- **Outcome** — wrist angle change (degrees), per-finger flexion change (degrees per joint), peak latency, movement type.
- **Context** — synchronized audio recording of the session, automatically transcribed by Whisper.

---

## 2. Hardware Setup

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   📱 iPhone #1                  💻 Laptop webcam        │
│   (HAND_CAMERA = 1)             (POSE_CAMERA = 2)       │
│   zoomed on fingers             sees full upper body    │
│                                                         │
│       ▼                              ▼                  │
│   ┌─────────┐                   ┌─────────┐             │
│   │ Fingers │                   │  Wrist  │             │
│   │ angles  │                   │  angle  │             │
│   └─────────┘                   └─────────┘             │
│                                                         │
│   📱 iPhone #2 (FOREARM_CAMERA = 0)                     │
│   top-down on a printed mat                             │
│       ▼                                                 │
│   ┌──────────────────────────────────┐                  │
│   │ ArUco markers track:             │                  │
│   │  - Mat corners (IDs 0,2,3,4)     │                  │
│   │  - Wrist (ID 1)                  │                  │
│   │  - Elbow (ID 7)                  │                  │
│   │  - Electrodes (IDs 5, 6)         │                  │
│   └──────────────────────────────────┘                  │
│                                                         │
│   🔌 EMS Device (e.g., P24)                             │
│   8 channels, 0-100mA, up to 500μs pulse width          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### The printed mat

A flat surface with four ArUco markers fixed at known corner positions. From `config/settings.py`:

- Mat dimensions: **199mm × 380mm**
- Marker size: **16mm × 16mm**
- Corner IDs (clockwise from top-left): `[0, 2, 3, 4]`

The four corner markers let the system compute a **homography** — a perspective transform that maps any camera pixel to a real-world millimeter coordinate on the mat. This corrects for camera tilt automatically.

### ArUco marker ID assignments

| ID  | Purpose                            |
|-----|------------------------------------|
| 0   | Mat top-left corner                |
| 2   | Mat top-right corner               |
| 3   | Mat bottom-right corner            |
| 4   | Mat bottom-left corner             |
| 1   | Wrist (placed on participant)      |
| 7   | Elbow (placed on participant)      |
| 5, 6 | Electrode pads                    |

---

## 3. Architecture Overview

The code is organized into four layers:

```
┌──────────────────────────────────────────────────────────┐
│                   UI LAYER (PyQt6)                       │
│   main_window  ·  graphics_view  ·  sign_in_dialog       │
│   parameter_panel  ·  stimulation_panel  ·  previews     │
└──────────────────────────────────────────────────────────┘
                            ↕
┌──────────────────────────────────────────────────────────┐
│                    CORE LAYER                            │
│   EMSController     — talks to the EMS device            │
│   PoseTracker       — wrist angle (laptop cam)           │
│   HandTracker       — finger angles (iPhone cam)         │
│   ForearmCamera     — mat + ArUco + electrodes           │
│   SessionRecorder   — audio to WAV                       │
│   SessionTranscriber— WAV to text (Whisper)              │
└──────────────────────────────────────────────────────────┘
                            ↕
┌──────────────────────────────────────────────────────────┐
│                    DATA LAYER                            │
│   EMSDatabase (SQLite)                                   │
│   • participants    • sessions          • electrodes     │
│   • stimulations    • movement_results  • joint_angles   │
│   • session_recordings  • session_transcriptions         │
└──────────────────────────────────────────────────────────┘
                            ↕
┌──────────────────────────────────────────────────────────┐
│                  CONFIG / SETTINGS                       │
│   config/settings.py — mat size, marker IDs, defaults    │
└──────────────────────────────────────────────────────────┘
```

### Three-layer ArUco vision pipeline

The forearm camera uses a clean 3-stage tracking pipeline (one frame at a time):

```
   Camera frame
        │
        ▼
┌─────────────────────┐
│  Layer 1            │   Detects 4 mat corner markers.
│  MatCalibrator      │   Computes homography (pixel ↔ mm).
└─────────────────────┘   Output: H matrix, px_per_mm
        │
        ▼
┌─────────────────────┐
│  Layer 2            │   Detects wrist (ID 1) and elbow (ID 7).
│  ArmTracker         │   Defines arm axis from the two markers.
└─────────────────────┘   Extrapolates s=1 from arm length in DB.
        │
        ▼
┌─────────────────────┐
│  Layer 3            │   Detects electrode markers (IDs 5, 6).
│  ElectrodeTracker   │   Converts pixels to (s, t) coordinates.
└─────────────────────┘   Resolves "place here" suggestions.
```

Each layer's output feeds the next. If a layer can't lock on (e.g., a mat corner is hidden), all downstream layers skip gracefully and the UI shows a clear status message.

---

## 4. The Full Workflow (Step-by-Step)

### Step 1 — App startup

```
$ python3 main.py
```

`main.py` fixes the Qt plugin path (a common macOS issue), then creates the `QApplication` and shows `EMSWindow`. The window immediately pops up the **sign-in dialog** before anything else happens.

### Step 2 — Sign in

`SignInDialog` (in `ui/widgets/sign_in_dialog.py`) shows two pages:

- **Page 1**: Dropdown of existing participants pulled from the database, with a preview showing age, arm dimensions, skin resistance values, and pain threshold.
- **Page 2**: New-participant form. Required: name + participant ID. Optional: age, arm width, arm length (wrist→elbow), skin resistance at 100 Hz / 1 kHz / 10 kHz / 100 kHz, experience level.

When the user clicks "Continue", the participant data is stored in `self.participant` on the main window and shown in the blue header bar at the top of the app.

### Step 3 — Click "Start Session"

This is the moment everything wakes up. The order matters:

1. **Create session row in DB** — `db.create_session(...)` returns a `session_id` (e.g. 55) which becomes the anchor for everything that follows.
2. **Connect to EMS device** — `EMSController.connect()` uses autodetect to find the P24 over USB serial.
3. **Start three cameras** (each in its own background thread):
   - `ForearmCamera` (camera 0) — opens at 1920×1080, runs ArUco detection every frame.
   - `PoseTracker` (camera 2) — opens with MediaPipe Pose Landmarker.
   - `HandTracker` (camera 1) — opens with MediaPipe Hand Landmarker.
4. **Open two preview windows** — small floating windows showing the live pose/hand feeds with skeleton overlays.
5. **Start audio recording** — `SessionRecorder` opens the default mic, creates a WAV file at `data/recordings/session_{id}_{pid}_{timestamp}.wav`, and writes audio chunks straight to disk every ~93 ms (crash-safe).

The right-side panel now shows the live forearm feed with a green border, grid lines, and corner labels — this is the "calibration UI" — no clicks needed.

### Step 4 — Place electrodes

The user physically places the two electrodes (with ArUco markers 5 and 6 stuck on them) on the forearm. The forearm camera detects them in real time.

When the user clicks **"Place Electrodes"** (or presses **P**), the system validates:

| Check                        | Required ArUco IDs |
|------------------------------|--------------------|
| Mat corners detected         | 0, 2, 3, 4         |
| Wrist marker on participant  | 1                  |
| Both electrode pads          | 5 and 6            |

If anything is missing, a red status message tells the user exactly what's wrong (e.g. "Electrode markers missing: ID [6] — attach to electrode pads").

If all checks pass:

- Builds the `AnatomicalCoordinateMapper` from the ArUco state — a math object that converts any pixel to anatomical (s, t) coordinates.
- Stores each electrode's position relative to the wrist:
  - `along_mm` — distance toward elbow (positive)
  - `lateral_mm` — distance from arm centerline (+ = thumb side, − = pinky side)
- Enables the STIMULATE button.

### Step 5 — Set stimulation parameters

The `StimulationPanel` on the right has five fields:

| Field          | Default | Range          |
|----------------|---------|----------------|
| Channel        | 6       | 0–7            |
| Intensity      | 5 mA    | 0–100 mA       |
| Pulse Width    | 250 μs  | 0–500 μs       |
| Pulse Count    | 10      | 1–1000         |
| Delay          | 10 ms   | 0–10,000 ms    |

The user can adjust these by typing, or use hotkeys (**I**/**U** = ±1 mA, **W**/**Q** = ±10 μs, **C**/**X** = ±5 pulses, **R** = reset).

### Step 6 — Click "STIMULATE" (or press Space)

This is the most carefully orchestrated moment in the app. Here's exactly what happens:

```
T = 0 ms ─────────────────────────────────────────────────────
   │
   │  PoseTracker.capture_baseline()
   │     averages last 5 wrist-angle samples → −3.2°
   │     saves baseline snapshot to captures/pose_055/
   │
   │  HandTracker.capture_baseline()
   │     averages last 5 hand-landmark frames
   │     calculates per-finger flexion (thumb:45°, index:30°...)
   │     saves baseline snapshot to captures/hand_055/
   │
T = 30 ms ────────────────────────────────────────────────────
   │
   │  db.record_stimulation(session=55, channel=6, intensity=5,
   │                         pulse_width=250, pulse_count=10, delay=10)
   │     → returns stim_id = 142
   │
   │  Capture current forearm image → captures/forearm_055/stim_0142.jpg
   │
   │  Read live ArUco state → save each electrode position to DB:
   │     E5: dx=-12.3mm, dy=42.8mm from wrist
   │     E6: dx=  3.1mm, dy=51.4mm from wrist
   │
T = 50 ms ────────────────────────────────────────────────────
   │
   │  ┌─ Thread A (Pose) ──────────────────┐
   │  │ Record wrist angle for 1.6s        │
   │  │ Sample at 50 Hz (~80 frames)       │
   │  │ Find peak (max |delta|)            │
   │  │ Average frames in top 80% of peak  │
   │  └────────────────────────────────────┘
   │
   │  ┌─ Thread B (Hand) ──────────────────┐
   │  │ Record finger angles for 1.6s      │
   │  │ Sample at 50 Hz                    │
   │  │ Find peak, average top 80%         │
   │  └────────────────────────────────────┘
   │
   │  ┌─ Main thread ──────────────────────┐
   │  │ ems.continuous_stim(...)           │
   │  │ Fires 10 pulses × 10ms apart = 100ms total
   │  └────────────────────────────────────┘
   │
T = 1.65 s ───────────────────────────────────────────────────
   │
   │  Both threads return their results:
   │    wrist_result.wrist_angle_delta = +78.0° (extension)
   │    finger_result.primary_finger = "index" (45.2°)
   │
   │  _CombinedMovementResult bridges both into one object
   │
   │  db.record_movement(stim_id=142, combined)
   │     → INSERT INTO movement_results (...)
   │     → INSERT 15 rows INTO joint_angles (5 fingers × 3 joints)
   │
T = 1.7 s ────────────────────────────────────────────────────
   │
   │  Status bar shows:
   │  "Stimulation complete ✓ | Wrist: 78.0° ext |
   │   Fingers: index 45.2° | Peak at 340ms"
   │
   ▼
   Ready for next stimulation
```

The 80%-threshold averaging is important — instead of grabbing one peak frame (which is noisy), the trackers find the maximum displacement and then average every frame within 80% of that peak. This gives a stable, reproducible measurement.

### Step 7 — Repeat

The user typically performs many stimulations per session, adjusting electrode placement, channel, intensity, etc. Each one is logged independently with its own electrode snapshot.

### Step 8 — Click "Stop Session"

1. **Stop audio recording** — the WAV file is closed.
2. **Finalize recording row in DB** — duration and file size are stored.
3. **Trigger background transcription** — `SessionTranscriber` loads the Whisper "base" model (~1 GB RAM), transcribes the WAV to text, saves the text to `session_transcriptions`, and **deletes the WAV file** to save disk space.
4. **Stop all cameras** and close preview windows.
5. **Disconnect EMS device.**
6. **Update participant's pain threshold** — the highest intensity used in this session is compared with the participant's stored `pain_threshold_ma` and updated if higher.
7. **End session row** — `ended_at` timestamp is set.
8. Print database stats to the console.

### Step 9 — Export for analysis

Outside the app, the researcher can export everything to a single flat CSV:

```python
from data.database import EMSDatabase
db = EMSDatabase()
db.export_csv("training_data.csv")   # one row per stimulation, ~80 columns
```

The flat row contains everything an ML model needs: participant features, stimulation parameters, electrode positions, and movement outcomes. No joins required.

---

## 5. Component Deep Dive

### `EMSController` (`core/ems_controller.py`)

A thin wrapper around the third-party EMS library at `ems_lib/ems-main/ems/core.py`. Key methods:

- `connect(fast_mode=False)` — autodetects the device over USB serial.
- `stimulate(channel, intensity, pulse_width)` — single pulse.
- `continuous_stim(channel, intensity, pulse_width, pulse_count, delay)` — train of pulses, used in real experiments.

If the library isn't installed, the controller logs a warning and returns `False` for all calls — the app stays usable for testing UI without hardware.

### `PoseTracker` (`core/vision/pose_tracker.py`)

Tracks the **wrist angle** using MediaPipe Pose Landmarker on the laptop webcam.

The wrist angle is computed as a **signed angle**:
- `v1 = elbow → wrist` (forearm direction)
- `v2 = wrist → hand_center` where `hand_center = (index + pinky) / 2`
- `unsigned_angle = arccos(v1·v2 / |v1||v2|)`
- `deviation = 180° − unsigned_angle` (so a straight wrist ≈ 0°)
- Sign comes from the 2D cross product of v1 and v2:
  - **Positive** = extension (hand up)
  - **Negative** = flexion (hand down)
  - The image is mirrored (`cv2.flip`) so the sign math accounts for that.

Auto-arm-detection: if started with `arm="auto"`, every frame compares left vs right arm visibility scores from MediaPipe and switches to whichever side is more visible.

### `HandTracker` (`core/vision/hand_tracker.py`)

Tracks **per-finger joint angles** using MediaPipe Hand Landmarker. For each of 5 fingers it computes 3 joint angles (e.g. index → MCP, PIP, DIP). The flexion of a finger is the sum of its three joint angles — comparing baseline vs peak gives a `flexion_change` per finger.

### `ForearmCamera` (`core/vision/forearm_camera.py`)

The orchestrator that runs all three vision layers (`MatCalibrator` → `ArmTracker` → `ElectrodeTracker`) in a background thread. Each frame it produces:
- An **annotated frame** (grid, corner dots, contours, electrode circles) for display.
- A **state dictionary** containing the homography, scale, wrist/elbow positions, electrode positions, etc., consumed by `main_window` and `BodyDiagramView`.

### `Undistorter` (`core/vision/undistorter.py`)

Every camera has its own intrinsic calibration file at `calibration/camera_{index}.npz` produced by running `calibrate_cameras.py`. The `Undistorter` loads the camera matrix and distortion coefficients, precomputes remap maps once, then applies `cv2.remap` to every incoming frame in ~2 ms. Missing calibration files print a warning but the app continues with raw frames.

### `SessionRecorder` + `SessionTranscriber`

- **Recorder** uses PyAudio at 44.1 kHz mono 16-bit. Writes 4096-sample chunks directly to a WAV file every ~93 ms (so a crash loses at most one chunk). Auto-stops after 5 minutes.
- **Transcriber** uses OpenAI Whisper locally (default model: `base`). After successful transcription, the WAV file is deleted to save disk space; only the text remains in `session_transcriptions`.

### `BodyDiagramView` + Managers (`ui/graphics_view.py`, `ui/managers/`)

A `QGraphicsView` showing the live forearm feed plus interactive overlays:
- **SceneManager** — image, background, grid overlay (perspective-correct, rendered via the homography).
- **ElectrodeManager** — clickable electrode markers, channel tracking, pixel ↔ anatomical conversions.
- **CalibrationManager** — receives ArUco state and builds the coordinate mapper.

The grid is drawn live every frame using the inverse homography — so even if the camera is at an angle, the 10mm grid lines hug the physical mat perfectly.

---

## 6. Data Storage

All data lives in **one SQLite file** at `project_database/ems_data.db`. The schema is documented in `data/database.py`.

### Table structure

```
participants ──────────┐
                       │ 1:N
                       ▼
                   sessions ──────────────┐
                       │                  │
        ┌──────────────┼──────────────┐   │
        │ 1:N          │ 1:N          │   │ 1:N
        ▼              ▼              ▼   ▼
   stimulations   electrodes  session_recordings
        │              ▲              │
        │ 1:1          │ 1:N          │ 1:1
        ▼              │              ▼
movement_results       │      session_transcriptions
        │              │
        │ 1:N          │
        ▼              │
   joint_angles        │
                       │
   stimulations.stim_id is referenced by electrodes
   (saves the position of each electrode at the moment
    of each stimulation, so electrode "drift" is tracked)
```

### Key tables

| Table                      | Rows per session     | What it stores                                |
|----------------------------|----------------------|-----------------------------------------------|
| `participants`             | 1 per person, total  | demographics, arm size, resistance, threshold |
| `sessions`                 | 1                    | device, target gesture, calibration factor    |
| `electrodes`               | 2 per stimulation    | pixel + real-world position relative to wrist |
| `stimulations`             | N (one per fire)     | channel, intensity, pulse width, count, delay |
| `movement_results`         | 1 per stimulation    | wrist delta, finger flexion, peak latency     |
| `joint_angles`             | 15 per stimulation   | per-finger per-joint baseline + result + delta |
| `session_recordings`       | 1                    | WAV filepath, duration, status                |
| `session_transcriptions`   | 1                    | full transcribed text, language, model used   |

### The flat export

`db.export_flat_dataframe()` builds one row per stimulation by joining all these tables and pivoting the per-joint data into wide columns. Result: ~80 columns including `thumb_MCP_delta`, `index_PIP_delta`, `electrode_1_real_x_cm`, etc. — ready to feed into pandas/scikit-learn/PyTorch.

---

## 7. Coordinate Systems Explained

The system uses three different coordinate spaces, and it helps to keep them straight:

### Pixel space (camera frame)
- Integer x, y in raw camera pixels (e.g. 1920 × 1080).
- Origin at top-left.
- What OpenCV gives you directly.

### Mat space (real-world mm)
- Continuous x, y in millimeters.
- Origin at mat corner ID 0 (top-left of the printed mat).
- x increases right, y increases down.
- Computed from pixel space using the homography H (4-corner perspective transform).

### Anatomical space (s, t)
- Normalized, dimensionless.
- `s` = longitudinal position along the forearm: **0 = wrist, 1 = elbow**.
- `t` = lateral position perpendicular to the forearm axis, normalized by forearm width: **0 = centerline, +1 = thumb/radial side, −1 = pinky/ulnar side**.
- The same `(s, t)` on different participants describes the same anatomical landmark — this is what makes electrode positions transferable across people.

### Why three?

- **Pixel space** is for drawing on screen.
- **Mat space** is for sub-millimeter measurements that don't depend on camera angle.
- **Anatomical space** is for generalization: an electrode at `(s=0.4, t=0.1)` will land on roughly the same muscle whether the participant has a 24 cm or 30 cm forearm.

The system stores all three for every electrode placement.

---

## 8. File Map

```
ems_lib/ems-main/ems/app_dev/
│
├── main.py                          # entry point
├── requirements.txt
│
├── config/
│   └── settings.py                  # mat size, marker IDs, defaults, channels
│
├── core/
│   ├── ems_controller.py            # EMS device wrapper
│   ├── session_recorder.py          # audio → WAV
│   ├── session_transcriber.py       # WAV → text (Whisper)
│   └── vision/
│       ├── mat_calibrator.py        # Layer 1: homography from 4 corners
│       ├── arm_tracker.py           # Layer 2: wrist + elbow + forearm axis
│       ├── electrode_tracker.py     # Layer 3: electrodes in (s,t)
│       ├── forearm_camera.py        # Orchestrates layers 1-3
│       ├── pose_tracker.py          # Wrist angle (laptop cam)
│       ├── hand_tracker.py          # Finger angles (iPhone cam)
│       ├── anatomical_mapper.py     # pixel ↔ (s,t) conversion
│       └── undistorter.py           # Lens calibration applier
│
├── data/
│   ├── database.py                  # SQLite schema + CRUD + export
│   ├── models.py                    # Session, Electrode, StimulationEvent
│   └── get_db_to_csv.py             # quick CSV export script
│
└── ui/
    ├── main_window.py               # EMSWindow — the big coordinator
    ├── graphics_view.py             # BodyDiagramView (the camera display)
    ├── managers/
    │   ├── scene_manager.py         # image + grid drawing
    │   ├── electrode_manager.py     # electrode placement logic
    │   ├── calibration_manager.py   # ArUco → mapper
    │   └── grid_overlay.py          # legacy fixed grid
    └── widgets/
        ├── sign_in_dialog.py        # participant selection
        ├── parameter_panel/         # device + start/stop
        ├── stimulation_panel.py     # channel, mA, μs, count, delay
        ├── electrode_placement_panel.py
        ├── electrode_marker.py      # the round numbered circles
        ├── forearm_preview_window.py
        └── tracking_preview.py      # floating pose/hand windows
```

---

## 9. Running the App

### First-time setup

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# macOS — install PortAudio for PyAudio
brew install portaudio
brew install ffmpeg               # needed by Whisper

# Calibrate each camera (one-time per camera, per laptop)
python3 calibrate_cameras.py --camera 0
python3 calibrate_cameras.py --camera 1
python3 calibrate_cameras.py --camera 2
```

### Launching

```bash
python3 main.py
```

### macOS Qt plugin error

If you see `Could not find the Qt platform plugin "cocoa"`, run:

```bash
export QT_QPA_PLATFORM_PLUGIN_PATH="$(python3 -c "import PyQt6, pathlib; print(pathlib.Path(PyQt6.__file__).parent/'Qt6'/'plugins'/'platforms')")"
python3 main.py
```

`main.py` already attempts this fix at startup, but the environment variable approach is the manual fallback.

---

## 10. Hotkeys & Tips

| Key       | Action                          |
|-----------|---------------------------------|
| **Space** | Trigger stimulation              |
| **P**     | Place electrodes / calibrate     |
| **I**     | Intensity +1 mA                  |
| **U**     | Intensity −1 mA                  |
| **W**     | Pulse width +10 μs               |
| **Q**     | Pulse width −10 μs               |
| **C**     | Pulse count +5                   |
| **X**     | Pulse count −5                   |
| **R**     | Reset all parameters to defaults |

Hotkeys are disabled while a text input has keyboard focus — so you can still type numbers without firing stimulations.

### Tips for a smooth session

- **Lighting matters** — ArUco detection is sensitive to glare. Diffuse, even lighting on the mat is ideal.
- **Don't move the mat camera mid-session** — the homography is recomputed every frame, so it'll adapt, but you'll get cleaner data with a fixed tripod.
- **Watch the floating preview windows** — if MediaPipe loses the hand or pose, baselines won't capture and you'll see warnings in the console. Adjust camera framing.
- **The ArUco status label below the body view tells the truth** — green text means all four mat corners are stable; orange means still acquiring; red means a marker is missing.
- **Recordings auto-stop at 5 minutes** — if your session needs to be longer, plan to stop and restart.

---

## Glossary

| Term              | Meaning                                                                                 |
|-------------------|-----------------------------------------------------------------------------------------|
| **EMS**           | Electrical Muscle Stimulation — small electrical pulses that make muscles contract.     |
| **ArUco marker**  | A black-and-white square barcode read by OpenCV to identify and locate physical objects. |
| **Homography**    | A 3×3 matrix that maps points between two planes — here, from camera image to mat surface. |
| **MediaPipe**     | Google's library for body/hand/face landmark detection. Used here for pose + hand.       |
| **Whisper**       | OpenAI's open-source speech-to-text model. Runs locally on your machine.                |
| **Baseline**      | The averaged pose/hand state captured *just before* stimulation, used as the reference. |
| **(s, t)**        | Anatomical coordinates — s along the arm (0=wrist, 1=elbow), t across it (thumb=+).     |
| **Pulse width**   | How long each electrical pulse lasts (in microseconds). Affects sensation and recruitment. |
| **Pulse count**   | Number of pulses in one stimulation burst. With 10ms delay, 10 pulses = 100ms total.    |
