"""
core/beam_detector.py
---------------------
Geometric Beam & Flag Detection for OMR — v2 (HoughLinesP + Diagonal Support)

This module uses classical computer vision (OpenCV) to determine the rhythmic
value of a note by analysing the area around its stem tip. It counts how many
beams (bars connecting stems) or flags (curly attachments on a single stem)
are present, allowing the pipeline to **correct** YOLO's duration classification
without retraining the model.

v2 Changes:
  - Primary detection via HoughLinesP — works at ANY angle (0°–45°)
  - Fallback to horizontal morphological opening for simple/clean scores
  - New function: find_all_noteheads_y() for chord contour splitting

Beam count → duration mapping:
    0 beams/flags → quarter note  (pătrime)
    1 beam/flag   → eighth note   (optimă)
    2 beams/flags → sixteenth note (șaisprezecime)
    3 beams/flags → thirty-second  (treizecidoime)
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


# Duration lookup table: beam_count → (MusicXML type, beats in 4/4)
BEAM_DURATION_MAP = {
    0: ('quarter',  1.0),
    1: ('eighth',   0.5),
    2: ('16th',     0.25),
    3: ('32nd',     0.125),
}


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def detect_beams_for_note(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    class_name: str,
    gray_image: np.ndarray,
    interline: float,
    notehead_y: float,
) -> int:
    """
    Analyse the stem-tip region of a detected note to count beams/flags.

    Uses HoughLinesP as primary detector (handles diagonal beams), with
    fallback to horizontal morphology for clean/simple scores.

    Only processes solid noteheads (Quarter-note, Eighth-note, Sixteenth-note).
    For hollow noteheads (Half-note, Full-note) there are no beams by definition.

    Args:
        x_center:    Bounding box center X (pixels).
        y_center:    Bounding box center Y (pixels).
        width:       Bounding box width (pixels).
        height:      Bounding box height (pixels).
        class_name:  YOLO class name string.
        gray_image:  Full grayscale image (H×W numpy array).
        interline:   Staff interline distance (pixels) — used for scaling.
        notehead_y:  Precise Y coordinate of the notehead center.

    Returns:
        Number of beams detected (0, 1, 2, or 3).
        Returns -1 if the note type is not applicable (hollow/whole notes)
        or if detection fails (caller should fall back to YOLO class).
    """
    # Only analyse solid noteheads
    SOLID_CLASSES = {'Quarter-note', 'Eighth-note', 'Sixteenth-note'}
    if class_name not in SOLID_CLASSES:
        return -1

    if interline <= 0:
        return -1

    img_h, img_w = gray_image.shape

    # --- Step 1: Determine stem direction (up or down) ---
    bbox_top = y_center - height / 2
    bbox_bot = y_center + height / 2
    notehead_in_bottom_half = (notehead_y - bbox_top) > (height / 2)
    stem_goes_up = notehead_in_bottom_half

    # --- Step 2: Extract ROI around stem tip ---
    roi_height = max(int(interline * 1.8), 20)
    roi_width = max(int(interline * 3.0), 40)

    if stem_goes_up:
        tip_y = int(bbox_top)
        roi_y1 = max(0, tip_y - int(roi_height * 0.3))
        roi_y2 = min(img_h, tip_y + int(roi_height * 0.7))
    else:
        tip_y = int(bbox_bot)
        roi_y1 = max(0, tip_y - int(roi_height * 0.7))
        roi_y2 = min(img_h, tip_y + int(roi_height * 0.3))

    roi_x1 = max(0, int(x_center - roi_width / 2))
    roi_x2 = min(img_w, int(x_center + roi_width / 2))

    if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
        return -1

    roi = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]
    stem_x_local = x_center - roi_x1

    # Binarize ROI
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Step 3: Remove the vertical stem ---
    binary_no_stem = _remove_stem(binary, interline)

    # --- Step 4: Try HoughLinesP first (handles diagonal beams) ---
    beam_count = _detect_beams_hough(binary_no_stem, interline, stem_x_local)

    if beam_count >= 0:
        return min(beam_count, 3)

    # --- Step 5: Fallback to horizontal morphology ---
    beam_count = _detect_beams_morphology(binary_no_stem, interline)

    return min(max(beam_count, 0), 3)


def find_all_noteheads_y(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    class_name: str,
    gray_image: np.ndarray,
    interline: float,
) -> List[float]:
    """
    Discover ALL notehead Y-coordinates inside a single bounding box.

    Used for "contour splitting" — when NMS merges multiple chord noteheads
    into a single tall bbox, this function finds each individual notehead.

    Args:
        x_center, y_center, width, height: Bounding box coordinates.
        class_name:  YOLO class name.
        gray_image:  Full grayscale image.
        interline:   Staff interline distance (pixels).

    Returns:
        List of Y-coordinates (one per detected notehead), sorted top-to-bottom.
        Returns [y_center] if only one notehead found or detection fails.
    """
    h, w = gray_image.shape
    x1 = max(0, int(x_center - width / 2))
    x2 = min(w, int(x_center + width / 2))
    y1 = max(0, int(y_center - height / 2))
    y2 = min(h, int(y_center + height / 2))

    if y2 <= y1 or x2 <= x1:
        return [y_center]

    box_roi = gray_image[y1:y2, x1:x2]
    _, binary = cv2.threshold(box_roi, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Fill hollow notes so they become solid blobs
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # Morphological opening: erase thin stems and staff lines, keep fat noteheads
    kernel_sz = max(3, int(min(width, height) * 0.12))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))
    cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

    final_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    if not final_contours:
        return [y_center]

    # Filter: keep contours with area > 30% of the largest
    areas = [cv2.contourArea(c) for c in final_contours]
    max_area = max(areas)
    min_area_threshold = max_area * 0.3

    noteheads_y = []
    for c, area in zip(final_contours, areas):
        if area >= min_area_threshold:
            _, cy, _, ch = cv2.boundingRect(c)
            noteheads_y.append(y1 + cy + ch / 2.0)

    # Sort top-to-bottom
    noteheads_y.sort()

    # De-duplicate: noteheads must be at least 0.4 interline apart
    if len(noteheads_y) > 1 and interline > 0:
        filtered = [noteheads_y[0]]
        for y in noteheads_y[1:]:
            if abs(y - filtered[-1]) > interline * 0.4:
                filtered.append(y)
        noteheads_y = filtered

    return noteheads_y if noteheads_y else [y_center]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------




def _remove_stem(binary: np.ndarray, interline: float) -> np.ndarray:
    """Remove the thin vertical stem from a binarized ROI."""
    roi_h, roi_w = binary.shape
    v_kernel_h = max(int(roi_h * 0.5), 8)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
    stem_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Dilate stem slightly for clean removal
    stem_dilated = cv2.dilate(
        stem_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        iterations=1
    )
    return cv2.subtract(binary, stem_dilated)


def _detect_beams_hough(
    roi_binary: np.ndarray,
    interline: float,
    stem_x_local: float,
) -> int:
    """
    Detect beams at ANY angle using Probabilistic Hough Line Transform.

    This is the primary detection method (v2). Works on horizontal AND
    diagonal beams up to ~50° inclination.

    Args:
        roi_binary:    Binarized ROI with stem removed (ink = 255).
        interline:     Staff interline distance (pixels).
        stem_x_local:  X coordinate of the stem within the ROI.

    Returns:
        beam_count (0–3), or -1 if detection is inconclusive (use fallback).
    """
    if roi_binary is None or roi_binary.size == 0:
        return -1

    # Edge detection for Hough
    edges = cv2.Canny(roi_binary, 50, 150)

    # HoughLinesP parameters — tuned for beam detection
    min_len = max(int(interline * 0.5), 8)
    max_gap = max(int(interline * 0.3), 4)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    if lines is None or len(lines) == 0:
        return -1  # Inconclusive — caller will use fallback

    # Filter lines by angle: keep only quasi-horizontal (beam-like)
    # Beams: 0°–50° (including diagonals)
    # Stems: ~90° (already removed, but filter just in case)
    beam_candidates = []
    roi_h, roi_w = roi_binary.shape

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        # Accept angles 0°–50° (beams, including steep ones)
        if angle < 50:
            line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Line must be reasonably long (at least 40% of interline)
            if line_len >= interline * 0.4:
                mid_y = (y1 + y2) / 2.0
                mid_x = (x1 + x2) / 2.0
                # Line should be in the vicinity of the stem
                if abs(mid_x - stem_x_local) < interline * 2.5:
                    beam_candidates.append({
                        'y_mid': mid_y,
                        'angle': angle,
                        'length': line_len,
                    })

    if not beam_candidates:
        return -1  # Inconclusive

    # Cluster candidates by Y-midpoint to count distinct beams
    beam_candidates.sort(key=lambda b: b['y_mid'])

    min_gap = interline * 0.25  # Minimum Y gap between distinct beams
    clusters = 1
    last_y = beam_candidates[0]['y_mid']

    for cand in beam_candidates[1:]:
        if abs(cand['y_mid'] - last_y) > min_gap:
            clusters += 1
        last_y = cand['y_mid']

    return min(clusters, 3)


def _detect_beams_morphology(binary_no_stem: np.ndarray, interline: float) -> int:
    """
    Fallback: detect beams using horizontal morphological opening.

    This is the v1 method — works well on perfectly horizontal beams
    but fails on diagonal ones. Kept as a safety net.

    Args:
        binary_no_stem: Binarized ROI with stem removed.
        interline:      Staff interline distance.

    Returns:
        beam_count (0–3).
    """
    h_kernel_w = max(int(interline * 0.6), 8)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
    horizontal_only = cv2.morphologyEx(binary_no_stem, cv2.MORPH_OPEN, h_kernel)

    # Row projection to count beams
    row_projection = np.sum(horizontal_only, axis=1) / 255.0
    min_ink = max(h_kernel_w * 0.5, 4)
    active_rows = row_projection > min_ink

    # Count clusters of active rows
    beam_count = _count_clusters(active_rows, min_gap=max(2, int(interline * 0.15)))
    return beam_count


def _count_clusters(mask: np.ndarray, min_gap: int = 2) -> int:
    """
    Count the number of distinct clusters of True values in a 1D boolean array.
    Clusters separated by fewer than `min_gap` False values are merged.
    """
    if not np.any(mask):
        return 0

    closed = mask.copy().astype(np.uint8)
    if min_gap > 1:
        kernel = np.ones(min_gap, dtype=np.uint8)
        closed = cv2.morphologyEx(
            closed.reshape(1, -1),
            cv2.MORPH_CLOSE,
            kernel.reshape(1, -1)
        ).flatten()

    clusters = 0
    in_cluster = False
    for val in closed:
        if val > 0 and not in_cluster:
            clusters += 1
            in_cluster = True
        elif val == 0:
            in_cluster = False

    return clusters
