"""
core/tie_detector.py
--------------------
Detects ties and slurs geometrically using OpenCV morphology.

A TIE connects two notes of the SAME pitch: the second note is not re-struck,
it just extends the first note's duration.

A SLUR connects notes of different pitches but is a phrasing mark only (no
duration change). We treat detected arcs conservatively: only merge durations
when the connected notes share the same pitch (tie semantics).

Detection strategy:
  1. Remove horizontal staff lines (they dominate the binary image)
  2. Erode vertically to kill note heads/stems/flags (tall objects)
  3. Keep only thin, horizontally elongated blobs → these are arc candidates
  4. An arc candidate is considered a tie if:
       - width > 2× interline  (long enough to span two notes)
       - height < interline    (thin)
       - aspect ratio > 3:1
       - y-center is within ±interline of a staff's bounding box
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


def detect_ties(
    gray_image: np.ndarray,
    staffs,               # List[StaffGroup]
    interline: float,
) -> List[Dict]:
    """
    Scan the page for tie/slur arcs.

    Returns a list of dicts:
      {
        'staff_id':  int (id(staff)),
        'x_start':   float,
        'x_end':     float,
        'y_center':  float,
      }
    """
    if gray_image is None or len(staffs) == 0:
        return []

    h, w = gray_image.shape

    # ── 1. Binarise ──────────────────────────────────────────────────────────
    _, binary = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY_INV)

    # ── 2. Remove staff lines (horizontal runs of ink) ─────────────────────
    horiz_len = max(5, int(interline * 3))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    no_lines = cv2.subtract(binary, lines_mask)

    # ── 3. Remove vertical structures (stems, barlines) ────────────────────
    vert_len = max(3, int(interline * 1.5))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vert_mask = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, vert_kernel)
    no_vert = cv2.subtract(no_lines, vert_mask)

    # ── 4. Remove note heads (small blobs, roughly square) ─────────────────
    # Erode with a vertical kernel to kill anything taller than ~half interline
    kill_tall_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(2, int(interline * 0.4)))
    )
    arcs_only = cv2.erode(no_vert, kill_tall_kernel, iterations=1)

    # ── 5. Dilate horizontally to connect broken arc segments ───────────────
    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, int(interline * 1.0)), 1)
    )
    arcs_connected = cv2.dilate(arcs_only, connect_kernel, iterations=1)

    # ── 6. Find contours of candidate arcs ─────────────────────────────────
    contours, _ = cv2.findContours(
        arcs_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_arc_width  = interline * 2.0   # arcs must span at least 2 interlines
    max_arc_height = interline * 0.9   # arcs are thin
    min_aspect     = 3.0               # width/height > 3

    ties = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < min_arc_width:
            continue
        if ch > max_arc_height:
            continue
        if ch == 0 or (cw / ch) < min_aspect:
            continue

        arc_y_center = y + ch / 2.0
        arc_x_start  = float(x)
        arc_x_end    = float(x + cw)

        # Assign to the closest staff
        best_staff = None
        best_dist  = float('inf')
        for staff in staffs:
            # Arc should be inside or just above/below the staff
            staff_mid = (staff.top + staff.bottom) / 2.0
            dist = abs(arc_y_center - staff_mid)
            margin = (staff.bottom - staff.top) * 0.8
            if dist < margin and dist < best_dist:
                best_dist  = dist
                best_staff = staff

        if best_staff is None:
            continue

        ties.append({
            'staff_id': id(best_staff),
            'x_start':  arc_x_start,
            'x_end':    arc_x_end,
            'y_center': arc_y_center,
        })

    if ties:
        print(f"[TieDetector] 🔗 Found {len(ties)} tie/slur arc(s).")

    return ties


def arcs_for_staff(ties: List[Dict], staff_id: int) -> List[Dict]:
    """Filter ties belonging to a specific staff."""
    return [t for t in ties if t['staff_id'] == staff_id]


def is_covered_by_arc(x_prev: float, x_curr: float, staff_arcs: List[Dict]) -> bool:
    """
    Return True if there is an arc whose horizontal span overlaps the interval
    [x_prev, x_curr], suggesting the two notes are tied.
    """
    for arc in staff_arcs:
        # The arc must START after the first note and END before (or at) the second note
        # Allow generous tolerance (±20% of arc width)
        arc_w = arc['x_end'] - arc['x_start']
        tol = arc_w * 0.25

        if arc['x_start'] >= x_prev - tol and arc['x_end'] <= x_curr + tol:
            return True
        # Also catch arcs that fully span the gap
        if arc['x_start'] <= x_prev + tol and arc['x_end'] >= x_curr - tol:
            return True

    return False
