# Deep Dive: ArUco Electrode Location Capture

A detailed walkthrough of how the EMS Control GUI turns raw camera pixels into precise electrode positions on a participant's forearm. This is the most carefully engineered subsystem in the application — the rest of the system depends on it being correct.

---

## Table of Contents

1. [The Physical Setup](#1-the-physical-setup)
2. [Why ArUco at All](#2-why-aruco-at-all)
3. [Layer 1 — Mat Calibration](#3-layer-1--mat-calibration-matcalibrator)
4. [Layer 2 — Arm Tracking](#4-layer-2--arm-tracking-armtracker)
5. [Layer 3 — Electrode Tracking](#5-layer-3--electrode-tracking-electrodetracker)
6. [The Full Coordinate Chain](#6-the-full-coordinate-chain)
7. [What Gets Saved — and Where the Two Paths Diverge](#7-what-gets-saved--and-where-the-two-paths-diverge)
8. [The "Place Electrodes" Validation Flow](#8-the-place-electrodes-validation-flow)
9. [Smoothing Budget — Latency vs. Stability](#9-smoothing-budget--latency-vs-stability)
10. [Camera Lens Calibration](#10-camera-lens-calibration)
11. [Live Debugging Aid](#11-live-debugging-aid)

---

## 1. The Physical Setup

Seven ArUco markers participate in capturing electrode locations, in three distinct roles:

```
                    ┌─────────────────────────┐
                    │ Camera looking down  ↓  │
                    └─────────────────────────┘

         ┌──────┐                          ┌──────┐
         │  0   │ ← TL                  TR │  2   │
         └──────┘                          └──────┘
              ┌─────────────────────────────┐
              │                             │
              │       ╔═══╗                 │
              │       ║ 1 ║  wrist          │
              │       ╚═══╝                 │
              │         │                   │
              │       ╔═══╗                 │
              │       ║ 5 ║  electrode      │
              │       ╚═══╝                 │  380 mm
              │       ╔═══╗                 │
              │       ║ 6 ║  electrode      │
              │       ╚═══╝                 │
              │         │                   │
              │       ╔═══╗                 │
              │       ║ 7 ║  elbow          │
              │       ╚═══╝                 │
              │                             │
              └─────────────────────────────┘
         ┌──────┐                          ┌──────┐
         │  4   │ ← BL                  BR │  3   │
         └──────┘                          └──────┘
              ←──────── 199 mm ─────────→

  All markers: DICT_4X4_50, physical size 16 × 16 mm
```

Marker IDs are assigned in `config/settings.py`:

```python
MAT_IDS       = [0, 2, 3, 4]   # corners, clockwise from top-left
WRIST_ID      = 1
ELBOW_ID      = 7
ELECTRODE_IDS = [5, 6]
```

The three roles:

| Role               | IDs           | Purpose                                                        |
|--------------------|---------------|----------------------------------------------------------------|
| Mat corners        | 0, 2, 3, 4    | Define a fixed reference plane in real-world millimeters.      |
| Anatomical anchors | 1 (wrist), 7 (elbow) | Locate the participant's arm within the mat frame.       |
| Electrode markers  | 5, 6          | Stuck on the electrode pads themselves.                        |

Every printed marker is **16 mm × 16 mm**. This number matters because it's how the system bootstraps its pixel-to-millimeter scale.

---

## 2. Why ArUco at All

The system needs to know where each electrode sits, in real millimeters, on a participant's forearm — consistent across cameras, sessions, and participants. There are roughly three ways to do this:

The first is a **manual click workflow** — point at the wrist, point at the elbow, point at the thumb side, then click each electrode. This is what the old `CalibrationManager` did. It works but it's slow, error-prone, and re-clicking between stimulations is tedious.

The second is **colored stickers** tracked with classical image processing. This is fast but fragile — lighting changes, similar colors elsewhere in the frame, and skin tones near the marker color all break it.

The third is **ArUco** — square fiducial markers with binary patterns that OpenCV decodes to a numeric ID. Each marker reports four exact corners in pixel space and gives you both position and orientation. Detection is robust under varied lighting, the markers are cheap to print, and the ID lets you bind each marker to a specific role (mat corner vs wrist vs electrode) without ambiguity.

The system uses **DICT_4X4_50**, OpenCV's dictionary of 50 distinct 4×4-bit markers. The four-bit grid is detectable at the resolution and distance the experiment uses; the small dictionary size keeps confusion between markers low.

---

## 3. Layer 1 — Mat Calibration (`MatCalibrator`)

Every frame from the forearm camera goes through this layer first. The job is to find the four corner markers and compute a **homography** — a 3×3 matrix that maps any camera pixel to a real-world millimeter coordinate on the mat surface.

### Detection and smoothing

The detection loop is straightforward: convert the frame to grayscale, run `aruco.ArucoDetector.detectMarkers`, then for each detected marker check its ID against the mat corner list. The center pixel of each marker (averaged from its four corners) gets pushed into a per-marker rolling buffer:

```python
self._buffers = {mid: deque(maxlen=SMOOTH_FRAMES) for mid in MAT_IDS}
# SMOOTH_FRAMES = 15
```

Smoothing isn't optional — raw ArUco corner detections jitter by ±2 pixels even on a stationary marker, and that jitter would propagate into the homography and then into every electrode position. Fifteen frames at 30 fps gives half a second of averaging, which kills the jitter without adding noticeable lag (mat corners don't move during a session).

### Computing the homography

Once all four corner buffers have data, the system has four points in camera space and four corresponding points in mat space:

```python
MAT_CORNERS_MM = np.array([
    [0,     0  ],   # ID 0 → top-left
    [199,   0  ],   # ID 2 → top-right
    [199,   380],   # ID 3 → bottom-right
    [0,     380],   # ID 4 → bottom-left
], dtype=np.float32)
```

Four point correspondences are exactly enough to solve for a homography (eight degrees of freedom, four points × two coordinates = eight equations). `cv2.findHomography(src_pts, MAT_CORNERS_MM)` returns the 3×3 matrix **H**. From then on:

```python
mm_point = cv2.perspectiveTransform(pixel_point, H)
```

…gives the mat-mm coordinate of any pixel, and `cv2.perspectiveTransform(mm_point, inv(H))` goes the other direction. This is what corrects camera tilt — the mat doesn't have to be perpendicular to the camera, the homography absorbs the perspective.

### The `px_per_mm` scale factor

This is computed separately. Each corner marker, being 16 mm physically, projects to some pixel size in the image. The detector returns all four corners of each marker, so the system averages the four side lengths of each marker, then averages across all four detected mat markers:

```python
avg_side_px = mean([mean([side1, side2, side3, side4]) for marker in mat_markers])
px_per_mm = avg_side_px / 16.0
```

This gives one scalar that represents the camera's resolving power on the mat surface. Strictly speaking `px_per_mm` varies across the image when there's perspective — a marker at the far edge covers fewer pixels than a marker close to the camera. The averaged value is good enough because the system uses the **homography** (not `px_per_mm` alone) for precise position math; `px_per_mm` is used only for displaying scale information and as a rough conversion factor where sub-pixel accuracy isn't needed.

### Optional override

If `calibration/spatial_scale.json` exists, its `corrected_px_per_mm` value is loaded at startup and replaces the marker-derived value. This is for cases where the user has run a separate spatial calibration procedure.

---

## 4. Layer 2 — Arm Tracking (`ArmTracker`)

The mat is now mapped, but the system still doesn't know where the arm is. Layer 2 fixes that by detecting the **wrist marker (ID 1)** and **elbow marker (ID 7)** — two markers placed by the experimenter on the participant.

Both markers get their own **10-frame** smoothing buffers. The window is shorter than the mat's 15 frames because the wrist and elbow can shift slightly during a session and you want the tracker to follow.

### Computing the arm axes

Once both buffers have at least one sample, the arm axis is defined:

```python
diff = elbow_marker_px - wrist_px
arm_dir = diff / np.linalg.norm(diff)            # unit vector wrist → elbow
perp_dir = np.array([-arm_dir[1], arm_dir[0]])   # rotate 90° in image space
```

- `arm_dir` — the longitudinal axis, pointing from wrist toward elbow in camera pixels.
- `perp_dir` — perpendicular to the arm axis, conventionally chosen so positive values are on the thumb/radial side.

### Why arm length comes from the database, not the elbow marker

The elbow marker's position **does not** become the `s = 1` point in anatomical coordinates. Instead, the system uses the participant's **arm length** (entered at sign-in, stored in the `participants` table) to extrapolate where `s = 1` should be:

```python
arm_length_px = arm_length_mm * px_per_mm
elbow_anatomical_px = wrist_px + arm_dir * arm_length_px
```

The elbow marker only defines the *direction* of the arm — not the *length*. This decoupling means the experimenter can stick the elbow marker anywhere along the upper forearm and it still works, as long as it's roughly along the arm's axis. The actual `s = 1` point is computed from the measured arm length, which is a property of the participant, not the marker.

### Forearm width profile

This layer also runs an **arm contour detector** that segments the forearm from the mat background using the LAB color space's `a` channel (which separates skin tones from typical mat colors) plus Otsu thresholding and morphological closing. The contour is used to estimate forearm width as a function of position along the arm — a profile sampled at 20 points from `s=0` to `s=1`. This width profile becomes important in Layer 3 for computing the `t` coordinate.

---

## 5. Layer 3 — Electrode Tracking (`ElectrodeTracker`)

This layer detects markers **IDs 5 and 6** with **8-frame** smoothing — the shortest buffer, because electrodes are the things most likely to be deliberately moved.

### Computing `(s, t)` anatomical coordinates

For each detected electrode, the tracker computes anatomical `(s, t)` coordinates using the arm geometry from Layer 2:

```python
relative   = electrode_px - wrist_px
along_px   = dot(relative, arm_dir)            # projection onto arm axis
lateral_px = dot(relative, perp_dir)           # projection onto perpendicular

s = along_px / arm_length_px                   # 0 at wrist, 1 at elbow
width_at_s_mm = interpolate(s, width_profile)
t = lateral_px / (width_at_s_mm * px_per_mm / 2)   # normalized by half-width
```

The dot products turn the 2D electrode position into two scalars — distance along the arm and distance perpendicular to it. Dividing the longitudinal distance by `arm_length_px` makes `s` a normalized 0-to-1 fraction. Dividing the lateral distance by the half-width *at that specific s value* gives a `t` that runs roughly from −1 (pinky edge) to +1 (thumb edge) regardless of where along the arm you are.

### Why width-normalized `t` matters

The width-normalized `t` is what makes positions transferable across participants. An electrode at `(s=0.4, t=0.1)` on a thin forearm and on a thick forearm describes the same anatomical landmark — slightly thumb-side of center, 40% of the way from wrist to elbow — because the normalization scales out the absolute width difference.

### Fallback when contour detection fails

If the arm contour detection fails (low contrast, weird lighting, sleeve covering the arm), the system falls back to a rough estimate:

```python
t = lateral_px / (arm_length_px * 0.15)
```

…treating the arm as if its half-width were 15% of its length. This is crude but keeps the tracker functional.

---

## 6. The Full Coordinate Chain

A single electrode position passes through this transformation chain every frame:

```
Raw camera frame (1920 × 1080 BGR pixels)
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │ Undistort                              │  ── camera-specific ──
   │   cv2.remap, per-camera intrinsics     │
   └────────────────────────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │ Detect markers                         │
   │   aruco.detectMarkers, DICT_4X4_50     │
   └────────────────────────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │ Smooth                                 │
   │   8-frame rolling average per marker   │
   └────────────────────────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │ Apply homography H                     │  ── mat-relative ──
   │   camera pixel → mat mm                │
   └────────────────────────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │ Subtract wrist position                │
   │   → (dx_mm, dy_mm) in mat frame        │
   └────────────────────────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │ Project onto arm axis                  │  ── arm-relative ──
   │   along_mm = dot(rel, arm_dir)/ppmm    │
   │   lateral_mm = dot(rel, perp_dir)/ppmm │
   └────────────────────────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │ Normalize to (s, t)                    │  ── anatomical ──
   │   s = along / arm_length               │
   │   t = lateral / half_width_at_s        │
   └────────────────────────────────────────┘
```

The four bands distinguish *what each step depends on*:

- **Camera-specific** steps depend only on the camera and its calibration — they remove distortion and find marker pixels.
- **Mat-relative** steps depend on the mat geometry — the homography is built from the mat corners.
- **Arm-relative** steps depend on the participant's arm — wrist position and elbow direction.
- **Anatomical** steps depend on the participant's arm length and forearm width profile.

This separation is why the system is robust to the camera moving mid-session (the mat corners just re-detect and rebuild the homography) but is not robust to the wrist marker moving relative to the actual wrist (because the arm-relative steps trust that marker as ground truth for the wrist position).

---

## 7. What Gets Saved — and Where the Two Paths Diverge

Here's the surprising part: **the system writes electrode positions to the database at two different moments, using two different coordinate systems, and only one of them produces the anatomical (s, t).**

### Path 1 — "Place Electrodes" handler

When the user clicks **"Place Electrodes"** (or presses P), the `_on_place_electrodes` handler in `main_window.py` validates that all required markers are visible, then computes each electrode's position in the **arm frame** — distance along the arm axis and perpendicular to it:

```python
relative   = e_px - wrist_px
along_mm   = dot(relative, arm_dir)  / px_per_mm   # toward elbow
lateral_mm = dot(relative, perp_dir) / px_per_mm   # thumb side positive
```

These values are cached in `self._electrode_placement` but **not written to the database** at this point. This handler exists purely to validate the setup and enable the STIMULATE button.

### Path 2 — Stimulate handler

Inside `on_stimulate_requested`, fired every time the user clicks STIMULATE, the system re-reads the live ArUco state after recording the stimulation row and saves each electrode's position **in the mat frame** — not the arm frame:

```python
def px_to_mat_mm(px_pt):
    return cv2.perspectiveTransform(px_pt, H)

wrist_mm = px_to_mat_mm(wrist_px)
for i, (eid, info) in enumerate(electrodes.items()):
    e_mm    = px_to_mat_mm(info["pixel"])
    diff_mm = e_mm - wrist_mm
    dx_mm   = float(diff_mm[0])   # mat-frame x offset
    dy_mm   = float(diff_mm[1])   # mat-frame y offset

    db.record_electrode(
        session_id=...,
        channel=i,
        pixel_x=e_px[0],
        pixel_y=e_px[1],
        stim_id=stim_id,            # links electrode to this specific stim
        real_x_cm=dx_mm / 10.0,
        real_y_cm=dy_mm / 10.0,
        placement_method="aruco_mat",
    )
```

### What ends up in the `electrodes` row

| Column                     | Filled by aruco path? | Meaning                                                       |
|----------------------------|------------------------|---------------------------------------------------------------|
| `pixel_x`, `pixel_y`       | yes                    | Raw camera pixels (useful only if camera doesn't move).       |
| `real_x_cm`, `real_y_cm`   | yes                    | **Mat-frame** offset from wrist marker, in centimeters.       |
| `stim_id`                  | yes                    | Link to the stimulation — position captured fresh every fire. |
| `placement_method`         | yes                    | Set to `"aruco_mat"`.                                         |
| `normalized_s`, `normalized_t` | **no**             | Computed by tracker but not persisted by this code path.      |
| `distance_from_wrist_cm`   | **no**                 | Schema field exists but not filled by this path.              |

### The two consequences

**Consequence 1 — positions are mat-frame, not arm-frame.** If the participant rotates their arm 30° relative to the mat between two sessions, an electrode at the same anatomical spot will have different `(real_x_cm, real_y_cm)` values. For analysis that wants pose-invariant positions, you'd need to recompute `(s, t)` from `pixel_x, pixel_y` plus the saved session-level wrist/elbow data — or modify the recording call to also persist `normalized_s` and `normalized_t` from `info["s"]` and `info["t"]`, which the tracker is already computing every frame.

**Consequence 2 — drift is tracked automatically.** Because positions are saved **with every stimulation**, not just once at placement time, the data captures exactly how the electrodes shifted across a session. The `electrodes` table will have 40 rows (2 electrodes × 20 stimulations) for a session with 20 stimulations — not just 2 rows. This is intentional, and lets later analysis correlate movement outcomes with subtle electrode-position changes.

---

## 8. The "Place Electrodes" Validation Flow

When the button is pressed, the system runs three gating checks against the live ArUco state. Each failure produces a specific red status message:

| Check                  | Required IDs   | Failure message                                                        |
|------------------------|----------------|------------------------------------------------------------------------|
| Mat corners visible    | 0, 2, 3, 4     | "Mat corners not detected — ensure IDs 0,2,3,4 are visible."           |
| Wrist marker visible   | 1              | "Wrist marker (ID=1) not detected — place it on the wrist."            |
| Both electrodes visible| 5 and 6        | "Electrode markers missing: ID [n] — attach to electrode pads."        |

Only when all three pass does the system build the `AnatomicalCoordinateMapper`, compute the arm-frame coordinates, enable the STIMULATE button, and turn the status label green.

The validation reads the state dict that `ForearmCamera` produces continuously in its background thread — it doesn't run a special detection pass. Whatever the most recent frame showed is what gets validated. This is fast (microseconds) but means a single bad frame is enough to fail. In practice the smoothing buffers make this a non-issue: by the time a marker has been visible for ~8 frames it's in the buffer and the validation succeeds.

---

## 9. Smoothing Budget — Latency vs. Stability

Each layer chose its buffer size deliberately:

| Layer       | Buffer size | Reasoning                                                                  |
|-------------|-------------|----------------------------------------------------------------------------|
| Mat corners | 15 frames   | These don't move during a session. Long buffer = maximally stable homography. |
| Wrist, elbow| 10 frames   | Can shift slowly as participant adjusts. Medium buffer = stable but follows real movement. |
| Electrodes  | 8 frames    | Actively repositioned by the experimenter. Short buffer = tracks repositioning quickly. |

At 30 fps these correspond to roughly **500 ms, 330 ms, and 270 ms** of averaging respectively. If the camera runs at the targeted 60 fps, those halve. The numbers are tuned for the current setup; lowering the electrode buffer to 4 would make the live display feel more responsive at the cost of more jitter, and raising it to 15 would feel laggy when moving electrodes.

---

## 10. Camera Lens Calibration

One step happens *before* everything else: **lens undistortion**. Most webcams and phone cameras have noticeable barrel distortion at the edges of the frame — straight lines appear curved. If you compute a homography from corner markers whose detected pixel positions are themselves distorted, the homography will be subtly wrong in a way that's worst at the corners (where the mat corners actually are).

The `Undistorter` class loads a per-camera calibration file at `calibration/camera_{index}.npz` that contains the camera matrix and distortion coefficients. These come from running `calibrate_cameras.py` once per physical camera, which uses a printed checkerboard to estimate the lens parameters. The undistorter precomputes the remap maps once and then applies `cv2.remap` to every frame in about 2 ms. The undistorted frame is what gets fed to ArUco detection — straight lines stay straight, corner detections sit where they belong, the homography fits cleanly.

If the calibration file is missing, the system prints a warning but continues with raw frames — the homography will still work, just less accurately near the image edges.

---

## 11. Live Debugging Aid

The `ElectrodePositionMonitor` dialog (defined at the bottom of `main_window.py`) is a small always-on-top window that prints, every frame, the current pixel coords, mat-mm coords, and wrist-relative offsets for each electrode. It's the ground truth for what the system is currently seeing — and matches exactly what gets written to the DB when STIMULATE is pressed:

```
Wrist: px=(964, 412)  mat=(98.2, 142.7) mm
E5:  px=(987, 358)
     mat=(101.4, 113.9) mm
     from wrist: dx=+3.2mm  dy=-28.8mm  dist=29.0mm
E6:  px=(992, 401)
     mat=(102.1, 130.4) mm
     from wrist: dx=+3.9mm  dy=-12.3mm  dist=12.9mm
```

If a stimulation produces unexpected database values, this monitor is the first place to look — if the monitor matches the DB, the issue is the marker positions; if they don't match, it's a code path issue.

---

## Appendix — Quick Reference

### Marker IDs

| ID | Role             | Smoothing |
|----|------------------|-----------|
| 0  | Mat corner TL    | 15 frames |
| 2  | Mat corner TR    | 15 frames |
| 3  | Mat corner BR    | 15 frames |
| 4  | Mat corner BL    | 15 frames |
| 1  | Wrist            | 10 frames |
| 7  | Elbow            | 10 frames |
| 5  | Electrode A      |  8 frames |
| 6  | Electrode B      |  8 frames |

### Mat dimensions

- Width: 199 mm (x-axis, ID 0 → ID 2)
- Height: 380 mm (y-axis, ID 0 → ID 4)
- Origin: ID 0 (top-left corner)

### Coordinate systems used

| System          | Origin           | Axes                                          | Stored in DB?      |
|-----------------|------------------|-----------------------------------------------|--------------------|
| Camera pixel    | image top-left   | image x right, image y down                   | yes (`pixel_x/y`)  |
| Mat mm          | mat corner ID 0  | x right (→ID 2), y down (→ID 4)               | indirectly via subtract from wrist |
| Mat mm − wrist  | wrist marker     | same axes as mat                              | yes (`real_x_cm`, `real_y_cm`) |
| Arm mm          | wrist marker     | along arm (→ elbow), perpendicular (thumb+)   | computed but not persisted |
| Anatomical (s,t)| wrist            | s = 0–1 wrist→elbow, t = ±1 normalized        | computed but not persisted |

### Key files

- `config/settings.py` — marker IDs, mat dimensions, marker size
- `core/vision/mat_calibrator.py` — Layer 1
- `core/vision/arm_tracker.py` — Layer 2
- `core/vision/electrode_tracker.py` — Layer 3
- `core/vision/forearm_camera.py` — orchestrates all three layers
- `core/vision/anatomical_mapper.py` — pixel ↔ (s, t) conversion
- `core/vision/undistorter.py` — lens undistortion
- `ui/main_window.py` — `_on_place_electrodes` and `on_stimulate_requested` handlers
- `data/database.py` — `record_electrode` method and schema
