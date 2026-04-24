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
DOUBLE_WINK_WINDOW_SEC = 1.0

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

# Simulated telemetry (static display values)
SIM_SPEED_KNOTS = 312
SIM_ALT_FEET    = 18500
SIM_HEADING_DEG = 247
SIM_G_FORCE     = 2.1


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
    Returns (pitch_deg, yaw_deg, roll_deg) or None on failure.
    Pitch: + = looking up, - = looking down
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
    # Decompose rotation matrix to Euler angles
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(-rmat[2, 1], rmat[2, 2]))
        yaw   = math.degrees(math.atan2( rmat[2, 0], sy))
        roll  = math.degrees(math.atan2(-rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = math.degrees(math.atan2( rmat[2, 0], sy))
        roll  = 0.0

    return pitch, yaw, roll


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

    Conceptual model
    ----------------
    The pitch card is fixed to the aircraft, not the camera.  The aircraft
    symbol and bank arc are painted on the glass (fixed).  The horizon and
    pitch rungs float behind the glass and move with the aircraft's attitude.

    Sign conventions (matching a real PFD):
      roll  > 0 → aircraft banks RIGHT  → horizon tilts LEFT on screen
      pitch > 0 → aircraft nose UP      → horizon moves DOWN on screen
      +N° rung  → above the horizon line
      -N° rung  → below the horizon line

    Implementation
    --------------
    card_pt(along, perp) maps a point in the tilted card's local frame to
    screen pixels.  'along' is parallel to the horizon; 'perp' is the card's
    local up-axis (positive = up on the card = up in the world).

    Roll is negated before building roll_rad so that a positive (right) roll
    rotates the horizon counter-clockwise on screen — exactly what you see
    from the cockpit as the world tilts left when banking right.

    Pitch moves the horizon DOWN when positive: the horizon perpendicular
    offset is +pitch*px_per_deg (positive perp = up on card = downward shift
    in screen-y because screen-y increases downward).
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    px_per_deg = h / 60.0

    # Negate roll: tilt right → horizon rotates counter-clockwise on screen
    roll_rad = math.radians(-roll)

    def card_pt(along: float, perp: float) -> tuple[int, int]:
        """
        along : left(-) / right(+) along the horizon
        perp  : down(-) / up(+) perpendicular to horizon (card local frame)
        Screen-y increases downward, so card-up maps to -screen-y before rotation.
        """
        # In screen space before rotation: right = +x, up = -y
        sx_local =  along
        sy_local = -perp          # card up → screen up (negative y)

        # Rotate by roll_rad (horizon tilt)
        dx = sx_local * math.cos(roll_rad) - sy_local * math.sin(roll_rad)
        dy = sx_local * math.sin(roll_rad) + sy_local * math.cos(roll_rad)

        # Pitch shifts the whole card along the card's up-axis.
        # Positive pitch → nose up → horizon moves DOWN → +screen-y shift
        pitch_dx = -pitch * px_per_deg * math.sin(roll_rad)
        pitch_dy =  pitch * px_per_deg * math.cos(roll_rad)

        return (int(cx + dx + pitch_dx), int(cy + dy + pitch_dy))

    # ── 0° horizon line ──────────────────────────────────────────────────────
    cv2.line(frame,
             card_pt(-w // 3, 0),
             card_pt( w // 3, 0),
             HUD_GREEN, 2, cv2.LINE_AA)

    # ── Pitch ladder rungs ───────────────────────────────────────────────────
    # Rungs are screen-horizontal (ignore roll) and only slide vertically
    # with pitch.  Only the horizon line itself tilts with roll.
    # Pitch shift in screen-y: nose up → horizon drops → rungs shift down too.
    pitch_screen_y = int(pitch * px_per_deg)

    for deg in range(-20, 25, 5):
        if deg == 0:
            continue

        # Screen-y for this rung: above boresight for positive deg
        rung_y   = cy + pitch_screen_y - int(deg * px_per_deg)
        half_len = 35 if deg % 10 == 0 else 18
        gap      = 20

        cv2.line(frame, (cx - half_len, rung_y), (cx - gap, rung_y),
                 HUD_DIM, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, rung_y), (cx + half_len, rung_y),
                 HUD_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{deg:+d}", (cx + half_len + 4, rung_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, HUD_DIM, 1, cv2.LINE_AA)


def draw_bank_indicator(frame: np.ndarray, roll: float) -> None:
    """
    Draw bank angle indicator at top of frame.

    Fixed (glass-painted):
      - Semi-circular arc, open at the bottom
      - Tick marks at 0, ±10, ±20, ±30, ±45, ±60°

    Moving (driven by roll):
      - Small triangle whose BASE sits on the arc (outside) and whose
        APEX points inward toward the arc centre.
        Roll right → triangle moves right along the arc.

    Arc geometry:
      Centre of the arc circle is above the top of the frame so only the
      lower portion of the arc is visible — exactly like a PFD.
      0° bank = triangle at top-centre of the arc.
    """
    h, w   = frame.shape[:2]
    cx     = w // 2
    radius = 90
    cy_arc = radius + 12   # arc centre inside frame — arc is fully visible

    # ── Fixed arc ────────────────────────────────────────────────────────────
    cv2.ellipse(frame, (cx, cy_arc), (radius, radius),
                0, 210, 330, HUD_DIM, 1, cv2.LINE_AA)

    # ── Fixed tick marks (drawn inward from arc) ──────────────────────────────
    tick_angles_deg = [0, 10, 20, 30, 45, 60, -10, -20, -30, -45, -60]
    for t in tick_angles_deg:
        # 0° bank = top of arc.  Map to standard math angle: top = 90°.
        # Right bank (+t) moves clockwise on screen → subtract t from 90°.
        ang_rad  = math.radians(90 - t)
        ox = int(cx + radius * math.cos(ang_rad))
        oy = int(cy_arc - radius * math.sin(ang_rad))
        tick_len = 10 if t % 30 == 0 else 6
        ix = int(cx + (radius - tick_len) * math.cos(ang_rad))
        iy = int(cy_arc - (radius - tick_len) * math.sin(ang_rad))
        cv2.line(frame, (ox, oy), (ix, iy), HUD_DIM, 1, cv2.LINE_AA)

    # ── Moving triangle ───────────────────────────────────────────────────────
    # tip sits ON the arc; apex points INWARD toward cy_arc.
    tri_ang_rad = math.radians(90 - roll)
    tip_x = int(cx + radius * math.cos(tri_ang_rad))
    tip_y = int(cy_arc - radius * math.sin(tri_ang_rad))

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


def draw_heading_tape(frame: np.ndarray, yaw: float) -> None:
    """
    Draw heading indicator below the bank angle arc.
    Positioned at y=120 so it clears the arc (radius=90, centre at y=-2).
    """
    h, w    = frame.shape[:2]
    heading = int(SIM_HEADING_DEG + yaw) % 360
    text    = f"HDG  {heading:03d}"
    y_pos   = 120
    cv2.putText(frame, text, (w // 2 - 48, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_GREEN, 1, cv2.LINE_AA)
    # Small tick below the text pointing up toward the arc
    cv2.line(frame, (w // 2, y_pos + 4), (w // 2, y_pos + 12),
             HUD_GREEN, 1, cv2.LINE_AA)


def draw_telemetry(frame: np.ndarray, pitch: float,
                   yaw: float, roll: float) -> None:
    """Draw speed, altitude, G-force, and angle readouts in HUD corners."""
    h, w = frame.shape[:2]
    margin = 12
    line_h = 18

    def put(text: str, x: int, y: int, color=HUD_GREEN) -> None:
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # ── Left column ──
    put(f"SPD  {SIM_SPEED_KNOTS} kts",  margin, h - margin - line_h * 3)
    put(f"ALT  {SIM_ALT_FEET:,} ft",    margin, h - margin - line_h * 2)
    put(f"G    {SIM_G_FORCE:.1f}",       margin, h - margin - line_h)

    # ── Right column ──
    put(f"PITCH  {pitch:+.1f}°", w - 155, h - margin - line_h * 3)
    put(f"YAW    {yaw:+.1f}°",   w - 155, h - margin - line_h * 2)
    put(f"ROLL   {roll:+.1f}°",  w - 155, h - margin - line_h)

    # ── Top right: ammo ──
    put(f"AMMO  04", w - 110, 28, HUD_AMBER)


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

def run(args: argparse.Namespace) -> None:
    """Initialise cameras, MediaPipe, and run the main HUD loop."""

    world_cap = open_camera(args.world_cam,  "world camera")
    face_cap  = open_camera(args.face_cam,   "face camera")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,       # enables iris landmarks (478 total)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    wink_detector = WinkDetector(dominant_eye=args.dominant_eye)

    # State
    targets:      list[tuple[int, int]] = []
    selected_idx: int   = 0
    ammo:         int   = 4
    pitch:        float = 0.0
    yaw:          float = 0.0
    roll:         float = 0.0
    fire_flash_until: float = 0.0

    # Refresh targets every N world frames
    target_refresh_interval = 15
    world_frame_count       = 0

    print("[HUD] Running — press Q to quit.")
    print(f"[HUD] World cam index : {args.world_cam}")
    print(f"[HUD] Face cam index  : {args.face_cam}")
    print(f"[HUD] Dominant eye    : {args.dominant_eye}")

    while True:
        world_frame = read_frame(world_cap)
        face_frame  = read_frame(face_cap)

        if world_frame is None or face_frame is None:
            print("[HUD] Warning: dropped frame — skipping.")
            continue

        world_h, world_w = world_frame.shape[:2]
        face_h,  face_w  = face_frame.shape[:2]

        # ── Face tracking (face camera) ──────────────────────────
        face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_rgb.flags.writeable = False
        results = face_mesh.process(face_rgb)
        face_rgb.flags.writeable = True

        wink_event = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            pose = estimate_head_pose(landmarks, face_w, face_h)
            if pose is not None:
                pitch, yaw, roll = pose

            wink_event = wink_detector.update(landmarks, face_w, face_h)

        # ── Target detection (world camera, periodic) ─────────────
        world_frame_count += 1
        if world_frame_count % target_refresh_interval == 0:
            new_targets = detect_targets(world_frame)
            if new_targets:
                targets      = new_targets
                selected_idx = min(selected_idx, len(targets) - 1)

        # ── Gesture events ───────────────────────────────────────
        if wink_event == "single" and targets:
            selected_idx = (selected_idx + 1) % len(targets)

        elif wink_event == "double" and targets and ammo > 0:
            ammo            -= 1
            fire_flash_until = time.time() + 0.8

        # ── HUD rendering (world camera frame) ───────────────────
        draw_horizon(world_frame, pitch, roll)
        draw_bank_indicator(world_frame, roll)
        draw_heading_tape(world_frame, yaw)
        draw_boresight(world_frame)
        draw_targets(world_frame, targets, selected_idx,
                     fired=(time.time() < fire_flash_until))
        draw_telemetry(world_frame, pitch, yaw, roll)
        draw_fire_flash(world_frame, fire_flash_until)

        # ── Ammo depleted ─────────────────────────────────────────
        if ammo == 0:
            h, w = world_frame.shape[:2]
            cv2.putText(world_frame, "WINCHESTER", (w // 2 - 80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_RED, 2, cv2.LINE_AA)

        # ── Display ───────────────────────────────────────────────
        cv2.imshow("HUD — World View", world_frame)

        # Small face-cam debug window (optional, helps during setup)
        cv2.imshow("Face Tracking", cv2.resize(face_frame, (320, 240)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────
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
