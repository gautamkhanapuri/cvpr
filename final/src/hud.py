"""
hud.py — Fighter Jet AR HUD Simulator
======================================
Dual-camera AR heads-up display inspired by helmet-mounted sight systems.

Camera layout:
  - WORLD_CAM  : head-worn webcam; scene video + all HUD rendering
  - FACE_CAM   : laptop webcam; head pose estimation + blink/wink detection

Usage:
  python hud.py [--world-cam 0] [--face-cam 1] [--dominant-eye right]
  or via environment variables:
    WORLD_CAM_INDEX=0 FACE_CAM_INDEX=1 DOMINANT_EYE=right python hud.py

Controls:
  Q            : quit
  Single wink  : cycle to next target
  Double wink  : fire (within DOUBLE_WINK_WINDOW_SEC)
"""

import argparse
import os
import sys
import time
import math

import cv2
import mediapipe as mp
import numpy as np


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# MediaPipe FaceMesh landmark indices for left/right eyes (6 points each)
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Iris landmark indices (only available with refine_landmarks=True)
LEFT_IRIS_IDX  = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]

# solvePnP: 3-D model points of canonical face landmarks (in mm, origin = nose tip)
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip          (1)
    [0.0,   -330.0, -65.0  ],   # Chin              (152)
    [-225.0,  170.0, -135.0],   # Left eye corner   (263)
    [225.0,  170.0, -135.0 ],   # Right eye corner  (33)
    [-150.0, -150.0, -125.0],   # Left mouth corner (287)
    [150.0,  -150.0, -125.0],   # Right mouth corner(57)
], dtype=np.float64)

FACE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# EAR threshold below which the eye is considered closed
EAR_CLOSED_THRESHOLD   = 0.20
# Consecutive frames below threshold to register a blink
BLINK_CONSEC_FRAMES    = 2
# Seconds within which two blinks constitute a double-wink → FIRE
DOUBLE_WINK_WINDOW_SEC = 1.5

# Harris corner detector parameters
HARRIS_BLOCK_SIZE  = 3
HARRIS_KSIZE       = 3
HARRIS_K           = 0.04
HARRIS_THRESH_PERC = 0.15   # fraction of max response to threshold
MAX_TARGETS        = 5      # maximum simultaneous targets shown
MIN_TARGET_DIST    = 40     # minimum pixel distance between targets

# HUD colours (BGR)
HUD_GREEN      = (0,   255, 70)
HUD_RED        = (0,   50,  255)
HUD_AMBER      = (0,   200, 255)
HUD_WHITE      = (255, 255, 255)
HUD_DIM        = (0,   180, 50)

# Simulated telemetry — initial values (updated dynamically at runtime)
SIM_SPEED_INIT   = 312      # knots
SIM_ALT_INIT     = 18500    # feet
SIM_HEADING_INIT = 247      # degrees


# ──────────────────────────────────────────────
# Flight dynamics simulation
# ──────────────────────────────────────────────

class FlightSim:
    """
    Lightweight flight dynamics driven by head pose.

    Physics model (simplified, arcade-style):
      - Airspeed:  increases when pitched down, bleeds when pitched up.
                   Clamped to [120, 600] knots.
      - Altitude:  climbs when pitched up, descends when pitched down.
                   Clamped to [500, 60000] feet.
      - Heading:   turns in the direction of roll (coordinated turn model).
      - G-force:   1/cos(roll) base (bank contribution) plus a pitch-rate
                   component so aggressive head swings spike the G-meter.
                   Smoothed with a low-pass filter to avoid jitter.
                   Clamped to [1.0, 9.0] G.

    Call update(pitch_deg, roll_deg, dt) every frame.
    """

    def __init__(self) -> None:
        self.speed   = float(SIM_SPEED_INIT)
        self.alt     = float(SIM_ALT_INIT)
        self.heading = float(SIM_HEADING_INIT)
        self.g_force = 1.0

        self._prev_pitch: float | None = None
        self._g_smooth:   float        = 1.0

    def update(self, pitch: float, roll: float, dt: float) -> None:
        """
        pitch : calibrated pitch in degrees (+ = nose up)
        roll  : roll in degrees             (+ = right bank)
        dt    : seconds since last frame
        """
        if dt <= 0 or dt > 1.0:
            return   # ignore stale or first frame

        # ── Airspeed ──────────────────────────────────────────────────────────
        # Pitch up → bleed ~3 kts/s per degree above 0; pitch down → gain.
        speed_rate = -pitch * 3.0
        self.speed = max(120.0, min(600.0, self.speed + speed_rate * dt))

        # ── Altitude ──────────────────────────────────────────────────────────
        # Climb/descent rate proportional to pitch and current speed.
        # Rule of thumb: fpm ≈ speed_knots * sin(pitch) * 101.3
        climb_fpm  = self.speed * math.sin(math.radians(pitch)) * 101.3
        self.alt   = max(500.0, min(60000.0, self.alt + climb_fpm * dt / 60.0))

        # ── Heading (coordinated turn) ────────────────────────────────────────
        # Turn rate (deg/s) = (G * tan(bank)) / (speed_knots * 0.0295)
        # Simplified: just use tan(roll) scaled to feel right.
        turn_rate    = math.tan(math.radians(roll)) * (self.speed / 300.0) * 3.0
        self.heading = (self.heading + turn_rate * dt) % 360.0

        # ── G-force ───────────────────────────────────────────────────────────
        # Bank contribution: G = 1/cos(roll), capped at 9G
        bank_g = 1.0 / max(math.cos(math.radians(roll)), 0.12)

        # Pitch rate contribution: fast pitch changes spike G
        pitch_rate_g = 0.0
        if self._prev_pitch is not None:
            pitch_rate    = abs(pitch - self._prev_pitch) / dt   # deg/s
            pitch_rate_g  = min(pitch_rate * 0.05, 4.0)          # scale to max +4G
        self._prev_pitch = pitch

        raw_g          = min(bank_g + pitch_rate_g, 9.0)
        # Low-pass smooth (tau ≈ 0.3s) to avoid single-frame spikes
        alpha          = dt / (dt + 0.3)
        self._g_smooth = self._g_smooth + alpha * (raw_g - self._g_smooth)
        self.g_force   = max(1.0, self._g_smooth)

    @property
    def speed_int(self) -> int:
        return int(round(self.speed))

    @property
    def alt_int(self) -> int:
        return int(round(self.alt / 100) * 100)   # round to nearest 100 ft

    @property
    def heading_int(self) -> int:
        return int(round(self.heading)) % 360


# ──────────────────────────────────────────────
# Argument / environment parsing
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI args; fall back to environment variables."""
    parser = argparse.ArgumentParser(description="Fighter Jet AR HUD Simulator")
    parser.add_argument(
        "--world-cam", type=int,
        default=int(os.environ.get("WORLD_CAM_INDEX", 0)),
        help="Camera index for head-worn world camera (default: 0)"
    )
    parser.add_argument(
        "--face-cam", type=int,
        default=int(os.environ.get("FACE_CAM_INDEX", 1)),
        help="Camera index for laptop face-tracking camera (default: 1)"
    )
    parser.add_argument(
        "--dominant-eye", type=str,
        default=os.environ.get("DOMINANT_EYE", "right").lower(),
        choices=["left", "right"],
        help="Dominant eye used for wink gestures (default: right)"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────────────

def open_camera(index: int, label: str) -> cv2.VideoCapture:
    """Open a VideoCapture and raise if it fails."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {label} (index {index}). "
                           "Check --world-cam / --face-cam arguments.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def read_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Read one frame; return None on failure."""
    ret, frame = cap.read()
    return frame if ret else None


# ──────────────────────────────────────────────
# EAR — Eye Aspect Ratio
# ──────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices: list[int],
                     img_w: int, img_h: int) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 MediaPipe landmarks.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Reference: Soukupová & Čech, CVWW 2016.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * img_w, lm.y * img_h]))

    # Vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


# ──────────────────────────────────────────────
# Head pose estimation
# ──────────────────────────────────────────────

def build_camera_matrix(img_w: int, img_h: int) -> np.ndarray:
    """Approximate camera intrinsic matrix from image dimensions."""
    focal = img_w  # rough estimate: focal length ≈ image width
    cx, cy = img_w / 2.0, img_h / 2.0
    return np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1 ]
    ], dtype=np.float64)


def estimate_head_pose(landmarks, img_w: int, img_h: int):
    """
    Run solvePnP against 6 canonical face landmarks.
    Returns (pitch_deg, yaw_deg, roll_deg, rmat) or None on failure.

    Euler angles are used for telemetry readouts and flight simulation.
    rmat (3x3 rotation matrix) is used for horizon/bank drawing to avoid
    gimbal lock that destabilises Euler decomposition beyond ±40° yaw.

    Pitch: + = looking up,    - = looking down
    Yaw:   + = looking right, - = looking left
    Roll:  + = tilting right, - = tilting left
    """
    img_pts = []
    for lm_id in FACE_LANDMARK_IDS:
        lm = landmarks[lm_id]
        img_pts.append([lm.x * img_w, lm.y * img_h])
    img_pts = np.array(img_pts, dtype=np.float64)

    cam_mat  = build_camera_matrix(img_w, img_h)
    dist_cof = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL, img_pts, cam_mat, dist_cof,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)

    # Euler angles for telemetry / flight sim (gimbal lock above ~±70° yaw)
    sy       = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(-rmat[2, 1], rmat[2, 2]))
        yaw   = math.degrees(math.atan2( rmat[2, 0], sy))
        roll  = math.degrees(math.atan2(-rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = math.degrees(math.atan2( rmat[2, 0], sy))
        roll  = 0.0

    return pitch, yaw, roll, rmat


# ──────────────────────────────────────────────
# Target detection (Harris corners)
# ──────────────────────────────────────────────

def detect_targets(frame: np.ndarray) -> list[tuple[int, int]]:
    """
    Detect up to MAX_TARGETS salient scene points using Harris corner detector.
    Returns list of (x, y) pixel coordinates, well-separated by MIN_TARGET_DIST.
    """
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f32 = np.float32(gray)

    harris = cv2.cornerHarris(gray_f32, HARRIS_BLOCK_SIZE,
                               HARRIS_KSIZE, HARRIS_K)
    harris = cv2.dilate(harris, None)

    threshold = HARRIS_THRESH_PERC * harris.max()
    coords    = np.argwhere(harris > threshold)  # (row, col)

    if len(coords) == 0:
        return []

    # Sort by response strength (descending)
    responses = harris[coords[:, 0], coords[:, 1]]
    order     = np.argsort(responses)[::-1]
    coords    = coords[order]

    # Greedy non-maximum suppression by distance
    selected = []
    for (row, col) in coords:
        pt = (int(col), int(row))  # (x, y)
        too_close = any(
            math.hypot(pt[0] - s[0], pt[1] - s[1]) < MIN_TARGET_DIST
            for s in selected
        )
        if not too_close:
            selected.append(pt)
        if len(selected) >= MAX_TARGETS:
            break

    return selected


# ──────────────────────────────────────────────
# HUD drawing
# ──────────────────────────────────────────────

def draw_targets(frame: np.ndarray, targets: list[tuple[int, int]],
                 selected_idx: int, fired: bool) -> None:
    """Draw green circles on unselected targets; red reticle on selected."""
    for i, (x, y) in enumerate(targets):
        if i == selected_idx:
            color = HUD_RED
            # Concentric reticle circles
            for r in (18, 28, 38):
                cv2.circle(frame, (x, y), r, color, 1, cv2.LINE_AA)
            # Crosshair lines
            cv2.line(frame, (x - 45, y), (x - 20, y), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x + 20, y), (x + 45, y), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - 45), (x, y - 20), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y + 20), (x, y + 45), color, 1, cv2.LINE_AA)
            label = "LOCKED" if not fired else "SPLASH"
            cv2.putText(frame, label, (x + 42, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 12, HUD_GREEN, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y),  3, HUD_GREEN, -1)


def _rotated_pt(cx: int, cy: int, dx: float, dy: float,
                roll_rad: float) -> tuple[int, int]:
    """
    Rotate offset (dx, dy) by roll_rad around (cx, cy) and return pixel coord.
    This is the core helper that makes ALL pitch ladder rungs parallel to the
    horizon — every point is expressed in the tilted frame then rotated back
    to screen space.
    """
    cos_r = math.cos(roll_rad)
    sin_r = math.sin(roll_rad)
    sx = dx * cos_r - dy * sin_r
    sy = dx * sin_r + dy * cos_r
    return (int(cx + sx), int(cy + sy))


def draw_horizon(frame: np.ndarray, pitch: float, roll: float) -> None:
    """
    Draw artificial horizon line and pitch ladder.

    Simple direct geometry — no rmat, no gimbal lock risk:
      - Horizon line is centred on frame centre, rotated by -roll degrees
        (tilt right → horizon tilts left), shifted vertically by pitch.
      - Pitch ladder rungs are screen-horizontal, only sliding up/down.
      - px_per_deg controls sensitivity.

    Sign conventions (confirmed from live testing):
      pitch > 0 → looking up   → horizon shifts DOWN  (cy + pitch*scale)
      roll  > 0 → tilt right   → horizon rotates CCW  (angle = -roll)
    """
    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2

    px_per_deg  = h / 60.0
    roll_rad    = math.radians(-roll)          # negate: tilt right → CCW
    pitch_shift = int(pitch * px_per_deg)      # positive → shift down

    # Horizon centre shifts vertically with pitch
    horizon_cy = cy + pitch_shift

    # Horizon endpoints: rotate ±(w//3) around (cx, horizon_cy)
    half_len = w // 3
    cos_r    = math.cos(roll_rad)
    sin_r    = math.sin(roll_rad)
    lx = int(cx - half_len * cos_r)
    ly = int(horizon_cy - half_len * sin_r)
    rx = int(cx + half_len * cos_r)
    ry = int(horizon_cy + half_len * sin_r)
    cv2.line(frame, (lx, ly), (rx, ry), HUD_GREEN, 2, cv2.LINE_AA)

    # ── Pitch ladder rungs (screen-horizontal, slide with pitch only) ─────────
    for deg in range(-20, 25, 5):
        if deg == 0:
            continue
        rung_y   = cy + pitch_shift - int(deg * px_per_deg)
        half_len_r = 35 if deg % 10 == 0 else 18
        gap        = 20
        cv2.line(frame, (cx - half_len_r, rung_y), (cx - gap, rung_y),
                 HUD_DIM, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, rung_y), (cx + half_len_r, rung_y),
                 HUD_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{deg:+d}", (cx + half_len_r + 4, rung_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, HUD_DIM, 1, cv2.LINE_AA)


def draw_bank_indicator(frame: np.ndarray, roll: float) -> None:
    """
    Draw bank angle indicator using Euler roll angle directly.
    Tilt right (roll > 0) → triangle moves right along arc.
    """
    h, w   = frame.shape[:2]
    cx     = w // 2
    radius = 90
    cy_arc = radius + 12

    # ── Fixed arc ────────────────────────────────────────────────────────────
    cv2.ellipse(frame, (cx, cy_arc), (radius, radius),
                0, 210, 330, HUD_DIM, 1, cv2.LINE_AA)

    # ── Fixed tick marks ─────────────────────────────────────────────────────
    for t in [0, 10, 20, 30, 45, 60, -10, -20, -30, -45, -60]:
        ang_rad  = math.radians(90 - t)
        ox = int(cx + radius * math.cos(ang_rad))
        oy = int(cy_arc - radius * math.sin(ang_rad))
        tick_len = 10 if t % 30 == 0 else 6
        ix = int(cx + (radius - tick_len) * math.cos(ang_rad))
        iy = int(cy_arc - (radius - tick_len) * math.sin(ang_rad))
        cv2.line(frame, (ox, oy), (ix, iy), HUD_DIM, 1, cv2.LINE_AA)

    # ── Moving triangle ───────────────────────────────────────────────────────
    tri_ang_rad = math.radians(90 - roll)
    tip_x = int(cx + radius * math.cos(tri_ang_rad))
    tip_y = int(cy_arc - radius * math.sin(tri_ang_rad))

    perp_rad  = tri_ang_rad + math.pi / 2
    base_half = 6
    b1x = int(tip_x + base_half * math.cos(perp_rad))
    b1y = int(tip_y - base_half * math.sin(perp_rad))
    b2x = int(tip_x - base_half * math.cos(perp_rad))
    b2y = int(tip_y + base_half * math.sin(perp_rad))

    tri_height = 10
    inward_x = int(tip_x - tri_height * math.cos(tri_ang_rad))
    inward_y = int(tip_y + tri_height * math.sin(tri_ang_rad))

    pts = np.array([[b1x, b1y], [b2x, b2y], [inward_x, inward_y]], dtype=np.int32)
    cv2.fillPoly(frame,  [pts], HUD_GREEN)
    cv2.polylines(frame, [pts], True, HUD_GREEN, 1, cv2.LINE_AA)

    # Base is OUTSIDE the arc (away from centre) — two points perpendicular
    # to the radial direction at the tip
    perp_rad  = tri_ang_rad + math.pi / 2
    base_half = 6
    b1x = int(tip_x + base_half * math.cos(perp_rad))
    b1y = int(tip_y - base_half * math.sin(perp_rad))
    b2x = int(tip_x - base_half * math.cos(perp_rad))
    b2y = int(tip_y + base_half * math.sin(perp_rad))

    # Apex points INWARD (toward cy_arc) — flip the radial direction
    tri_height = 10
    inward_x = int(tip_x - tri_height * math.cos(tri_ang_rad))
    inward_y = int(tip_y + tri_height * math.sin(tri_ang_rad))

    pts = np.array([[b1x, b1y], [b2x, b2y], [inward_x, inward_y]], dtype=np.int32)
    cv2.fillPoly(frame,   [pts], HUD_GREEN)
    cv2.polylines(frame,  [pts], True, HUD_GREEN, 1, cv2.LINE_AA)


def draw_heading_tape(frame: np.ndarray, heading: int) -> None:
    """Draw heading indicator below the bank angle arc."""
    h, w   = frame.shape[:2]
    text   = f"HDG  {heading:03d}"
    y_pos  = 120
    cv2.putText(frame, text, (w // 2 - 48, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_GREEN, 1, cv2.LINE_AA)
    cv2.line(frame, (w // 2, y_pos + 4), (w // 2, y_pos + 12),
             HUD_GREEN, 1, cv2.LINE_AA)


def draw_telemetry(frame: np.ndarray, pitch: float, yaw: float, roll: float,
                   ammo: int, kills: int,
                   speed: int, alt: int, g_force: float) -> None:
    """Draw speed, altitude, G-force, angle readouts, ammo and kill count."""
    h, w = frame.shape[:2]
    margin = 12
    line_h = 18

    def put(text: str, x: int, y: int, color=HUD_GREEN) -> None:
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # ── Left column ──
    put(f"SPD  {speed} kts",   margin, h - margin - line_h * 3)
    put(f"ALT  {alt:,} ft",    margin, h - margin - line_h * 2)
    g_color = HUD_RED if g_force >= 7.0 else HUD_AMBER if g_force >= 5.0 else HUD_GREEN
    put(f"G    {g_force:.1f}", margin, h - margin - line_h, g_color)

    # ── Right column ──
    put(f"PITCH  {pitch:+.1f}°", w - 155, h - margin - line_h * 3)
    put(f"YAW    {yaw:+.1f}°",   w - 155, h - margin - line_h * 2)
    put(f"ROLL   {roll:+.1f}°",  w - 155, h - margin - line_h)

    # ── Top right: ammo and kills ──
    ammo_color = HUD_RED if ammo == 0 else HUD_AMBER
    put(f"AMMO  {ammo:02d}", w - 110, 28, ammo_color)
    put(f"KILLS {kills:02d}", w - 110, 46, HUD_GREEN)


def draw_fire_flash(frame: np.ndarray, flash_until: float) -> None:
    """Flash 'FOX TWO' text on screen for a short duration after firing."""
    if time.time() < flash_until:
        h, w = frame.shape[:2]
        cv2.putText(frame, "FOX TWO", (w // 2 - 60, h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, HUD_RED, 2, cv2.LINE_AA)


def draw_boresight(frame: np.ndarray) -> None:
    """
    Draw a fixed aircraft symbol at frame centre.
    Shape: a small V (fuselage/nose) with horizontal wings extending outward,
    plus a tiny centre dot.  This symbol is fixed to the glass — it never moves.

    Geometry (all offsets from centre cx, cy):
      V apex  : (cx, cy + v_depth)          ← nose points down on screen
      V left  : (cx - v_half, cy)
      V right : (cx + v_half, cy)
      Wing tips: (cx ± wing_len, cy)
      Wing-fuselage join: small vertical stub at each wing root
    """
    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2

    v_half   = 10    # half-width of the V
    v_depth  = 7     # how far down the apex of the V sits
    wing_len = 28    # total half-span of each wing from centre
    stub     = 4     # small vertical stub at wing root

    # V shape (nose)
    apex  = (cx,          cy + v_depth)
    v_lft = (cx - v_half, cy)
    v_rgt = (cx + v_half, cy)
    cv2.line(frame, v_lft, apex, HUD_WHITE, 1, cv2.LINE_AA)
    cv2.line(frame, v_rgt, apex, HUD_WHITE, 1, cv2.LINE_AA)

    # Left wing: from V left corner outward
    cv2.line(frame, (cx - v_half, cy), (cx - wing_len, cy),
             HUD_WHITE, 1, cv2.LINE_AA)
    # Left wing-root vertical stub
    cv2.line(frame, (cx - v_half, cy - stub), (cx - v_half, cy + stub),
             HUD_WHITE, 1, cv2.LINE_AA)

    # Right wing: from V right corner outward
    cv2.line(frame, (cx + v_half, cy), (cx + wing_len, cy),
             HUD_WHITE, 1, cv2.LINE_AA)
    # Right wing-root vertical stub
    cv2.line(frame, (cx + v_half, cy - stub), (cx + v_half, cy + stub),
             HUD_WHITE, 1, cv2.LINE_AA)

    # Centre dot
    cv2.circle(frame, (cx, cy), 2, HUD_WHITE, -1)


# ──────────────────────────────────────────────
# Blink / wink state machine
# ──────────────────────────────────────────────

class WinkDetector:
    """
    Tracks EAR per eye and fires single/double wink events.
    Only the dominant eye triggers target actions.
    """

    def __init__(self, dominant_eye: str) -> None:
        self.dominant_eye      = dominant_eye   # "left" or "right"
        self._consec_closed    = 0              # consecutive closed frames
        self._last_wink_time   = 0.0            # time of previous wink
        self._pending_single   = False          # waiting to confirm single vs double

    def update(self, landmarks, img_w: int, img_h: int) -> str | None:
        """
        Feed new landmarks. Returns:
          "single" — one wink detected
          "double" — double wink detected (FIRE)
          None     — no event
        """
        dom_indices = (RIGHT_EYE_IDX if self.dominant_eye == "right"
                       else LEFT_EYE_IDX)
        off_indices = (LEFT_EYE_IDX  if self.dominant_eye == "right"
                       else RIGHT_EYE_IDX)

        dom_ear = eye_aspect_ratio(landmarks, dom_indices, img_w, img_h)
        off_ear = eye_aspect_ratio(landmarks, off_indices, img_w, img_h)

        # A wink = dominant eye closed AND non-dominant open
        dom_closed = dom_ear < EAR_CLOSED_THRESHOLD
        off_open   = off_ear > EAR_CLOSED_THRESHOLD + 0.05

        if dom_closed and off_open:
            self._consec_closed += 1
        else:
            if self._consec_closed >= BLINK_CONSEC_FRAMES:
                # Eye just reopened — register a wink
                now = time.time()
                dt  = now - self._last_wink_time
                self._last_wink_time = now
                self._consec_closed  = 0

                if dt < DOUBLE_WINK_WINDOW_SEC:
                    return "double"
                return "single"
            self._consec_closed = 0

        return None


# ──────────────────────────────────────────────
# Main application loop
# ──────────────────────────────────────────────

def draw_face_mesh(frame: np.ndarray, landmarks, img_w: int, img_h: int) -> None:
    """Draw a minimal face mesh overlay on the face camera frame."""
    mp_drawing    = mp.solutions.drawing_utils
    mp_face_mesh_ = mp.solutions.face_mesh
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh_.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 200, 0), thickness=1, circle_radius=0
        )
    )


# ──────────────────────────────────────────────
# Optical flow tracker
# ──────────────────────────────────────────────

class OpticalFlowTracker:
    """
    Tracks a single scene point across frames using Lucas-Kanade optical flow.

    Interaction model:
      lock(frame, point)  — seed tracker on (x,y); samples nearby corners
      update(frame)       — propagate to next frame; returns centroid or None
      reset()             — stop tracking

    If fewer than MIN_POINTS survive an LK iteration, tracking is abandoned.
    """

    LK_PARAMS  = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                30, 0.01))
    MIN_POINTS = 3
    PATCH_R    = 20

    def __init__(self) -> None:
        self._pts:       np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self.active:     bool              = False

    def lock(self, frame: np.ndarray, point: tuple[int, int]) -> None:
        """Initialise tracking around (x, y)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y = point
        x1 = max(0, x - self.PATCH_R);  y1 = max(0, y - self.PATCH_R)
        x2 = min(frame.shape[1], x + self.PATCH_R)
        y2 = min(frame.shape[0], y + self.PATCH_R)
        roi = np.float32(gray[y1:y2, x1:x2])

        corners = None
        if roi.size > 0:
            corners = cv2.goodFeaturesToTrack(roi, maxCorners=10,
                                              qualityLevel=0.2, minDistance=5)
        if corners is None or len(corners) < self.MIN_POINTS:
            corners = np.array([[[float(x), float(y)]]], dtype=np.float32)
        else:
            corners[:, :, 0] += x1
            corners[:, :, 1] += y1

        self._pts       = corners
        self._prev_gray = gray
        self.active     = True

    def update(self, frame: np.ndarray) -> tuple[int, int] | None:
        """Propagate tracked points; return centroid or None if lost."""
        if not self.active or self._pts is None or self._prev_gray is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._pts, None, **self.LK_PARAMS
        )

        if new_pts is None or status is None:
            self.reset(); return None

        good = new_pts[status.flatten() == 1]
        if len(good) < self.MIN_POINTS:
            self.reset(); return None

        self._pts       = good.reshape(-1, 1, 2)
        self._prev_gray = gray
        good_2d = good.reshape(-1, 2)
        return (int(np.mean(good_2d[:, 0])), int(np.mean(good_2d[:, 1])))

    def reset(self) -> None:
        self._pts = None; self._prev_gray = None; self.active = False


def draw_tracked_target(frame: np.ndarray, pos: tuple[int, int],
                        fired: bool) -> None:
    """Animated tracking reticle — larger than the static Harris reticle."""
    x, y  = pos
    color = HUD_RED if not fired else (0, 0, 255)
    label = "TRACKING" if not fired else "SPLASH"
    for r in (22, 34, 46):
        cv2.circle(frame, (x, y), r, color, 1, cv2.LINE_AA)
    cv2.line(frame, (x - 55, y), (x - 25, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + 25, y), (x + 55, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - 55), (x, y - 25), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y + 25), (x, y + 55), color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x + 50, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> None:
    """Initialise cameras, MediaPipe, and run the main HUD loop."""

    world_cap = open_camera(args.world_cam, "world camera")
    face_cap  = open_camera(args.face_cam,  "face camera")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    wink_detector = WinkDetector(dominant_eye=args.dominant_eye)
    flight_sim    = FlightSim()
    tracker       = OpticalFlowTracker()

    # ── State ─────────────────────────────────────────────────────────────────
    targets:          list[tuple[int, int]]      = []
    selected_idx:     int                        = 0
    ammo:             int                        = 4
    kills:            int                        = 0
    pitch:            float                      = 0.0
    yaw:              float                      = 0.0
    roll:             float                      = 0.0
    rmat:             np.ndarray                 = np.eye(3)
    fire_flash_until: float                      = 0.0
    pitch_smooth:     float                      = 0.0
    roll_smooth:      float                      = 0.0
    last_frame_time:  float                      = time.time()
    tracked_pos:      tuple[int, int] | None     = None
    pitch_offset:     float                      = 0.0
    show_mirror:      bool                       = True
    show_face_mesh:   bool                       = False

    save_dir         = "hud_captures"
    os.makedirs(save_dir, exist_ok=True)
    screenshot_count = 0

    target_refresh_interval = 15
    world_frame_count       = 0

    print("[HUD] Running.")
    print("  Q — quit  |  M — mirror  |  F — face mesh  |  C — calibrate pitch")
    print("  S — save frame  |  L — lock/unlock tracker on selected target")
    print("  Wink       — cycle targets  (or unlock tracker if active)")
    print("  Double wink — lock tracker  (or FIRE if tracker active)")
    print(f"  World cam: {args.world_cam}  |  Face cam: {args.face_cam}  |  Eye: {args.dominant_eye}")

    while True:
        world_frame = read_frame(world_cap)
        face_frame  = read_frame(face_cap)

        if world_frame is None or face_frame is None:
            print("[HUD] Warning: dropped frame — skipping.")
            continue

        world_h, world_w = world_frame.shape[:2]
        face_h,  face_w  = face_frame.shape[:2]

        # ── Face tracking ─────────────────────────────────────────────────────
        face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_rgb.flags.writeable = False
        results = face_mesh.process(face_rgb)
        face_rgb.flags.writeable = True

        raw_landmarks = None
        wink_event    = None

        if results.multi_face_landmarks:
            raw_landmarks = results.multi_face_landmarks[0]
            landmarks     = raw_landmarks.landmark

            pose = estimate_head_pose(landmarks, face_w, face_h)
            if pose is not None:
                pitch, yaw, roll, rmat = pose
                if abs(roll) < 90.0:
                    alpha        = 0.25
                    pitch_smooth = pitch_smooth + alpha * (pitch - pitch_smooth)
                    roll_smooth  = roll_smooth  + alpha * (roll  - roll_smooth)
                    pitch_smooth = max(-30.0, min(30.0, pitch_smooth))

            wink_event = wink_detector.update(landmarks, face_w, face_h)

        calibrated_pitch = pitch_smooth - pitch_offset

        # ── Flight simulation ──────────────────────────────────────────────────
        now = time.time()
        dt  = now - last_frame_time
        last_frame_time = now
        flight_sim.update(calibrated_pitch, roll_smooth, dt)

        # ── Optical flow update ────────────────────────────────────────────────
        if tracker.active:
            tracked_pos = tracker.update(world_frame)
            if tracked_pos is None:
                print("[HUD] Tracking lost — reverting to Harris targets.")

        # ── Target detection (suspended while tracking) ───────────────────────
        world_frame_count += 1
        if not tracker.active and world_frame_count % target_refresh_interval == 0:
            new_targets = detect_targets(world_frame)
            if new_targets:
                targets      = new_targets
                selected_idx = min(selected_idx, len(targets) - 1)

        # ── Gesture events ────────────────────────────────────────────────────
        if wink_event == "single":
            if tracker.active:
                tracker.reset()
                tracked_pos = None
                print("[HUD] Tracker unlocked.")
            elif targets:
                selected_idx = (selected_idx + 1) % len(targets)

        elif wink_event == "double":
            if not tracker.active and targets:
                lock_pt = targets[selected_idx]
                tracker.lock(world_frame, lock_pt)
                tracked_pos = lock_pt
                print(f"[HUD] Tracker locked on {lock_pt}.")
            elif tracker.active and tracked_pos is not None and ammo > 0:
                ammo            -= 1
                kills           += 1
                fire_flash_until = time.time() + 1.2
                print(f"[HUD] FIRE — kills: {kills}  ammo: {ammo}")

        # ── HUD rendering ─────────────────────────────────────────────────────
        draw_horizon(world_frame, calibrated_pitch, roll_smooth)
        draw_bank_indicator(world_frame, roll_smooth)
        draw_heading_tape(world_frame, flight_sim.heading_int)
        draw_boresight(world_frame)

        if tracker.active and tracked_pos is not None:
            draw_tracked_target(world_frame, tracked_pos,
                                fired=(time.time() < fire_flash_until))
            cv2.putText(world_frame, "TRK", (12, world_h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_AMBER, 1, cv2.LINE_AA)
        else:
            draw_targets(world_frame, targets, selected_idx,
                         fired=(time.time() < fire_flash_until))

        draw_telemetry(world_frame, calibrated_pitch, yaw, roll_smooth, ammo, kills,
                       flight_sim.speed_int, flight_sim.alt_int, flight_sim.g_force)
        draw_fire_flash(world_frame, fire_flash_until)

        if ammo == 0:
            cv2.putText(world_frame, "WINCHESTER",
                        (world_w // 2 - 80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_RED, 2, cv2.LINE_AA)

        cv2.imshow("HUD — World View", world_frame)

        # ── Face mirror ────────────────────────────────────────────────────────
        if show_mirror:
            display_face = cv2.resize(face_frame, (320, 240))
            if show_face_mesh and raw_landmarks is not None:
                draw_face_mesh(display_face, raw_landmarks,
                               display_face.shape[1], display_face.shape[0])
            cv2.imshow("Face Tracking", display_face)
        else:
            try:
                cv2.destroyWindow("Face Tracking")
            except Exception:
                pass

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            show_mirror = not show_mirror
        elif key == ord("f"):
            show_face_mesh = not show_face_mesh
        elif key == ord("c"):
            pitch_offset = pitch_smooth
            print(f"[HUD] Pitch calibrated — offset set to {pitch_offset:+.1f}°")
        elif key == ord("s"):
            fname = os.path.join(save_dir, f"hud_{screenshot_count:04d}.png")
            cv2.imwrite(fname, world_frame)
            screenshot_count += 1
            print(f"[HUD] Saved {fname}")
        elif key == ord("l"):
            if tracker.active:
                tracker.reset()
                tracked_pos = None
                print("[HUD] Tracker unlocked.")
            elif targets:
                lock_pt = targets[selected_idx]
                tracker.lock(world_frame, lock_pt)
                tracked_pos = lock_pt
                print(f"[HUD] Tracker locked on {lock_pt}.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    face_mesh.close()
    world_cap.release()
    face_cap.release()
    cv2.destroyAllWindows()
    print("[HUD] Exited cleanly.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    try:
        run(args)
    except RuntimeError as e:
        print(f"[HUD] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
