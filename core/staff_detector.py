"""
core/staff_detector.py
-----------------------
Classical Computer Vision — Staff Line Detection

Detects the horizontal staff lines in a music sheet image using pure OpenCV
(no ML required). This module provides the spatial "grid" that the YOLO
detector needs to convert bounding-box Y positions into exact musical pitches.

Pipeline:
  1. Grayscale → Otsu binarization (invert so lines = white)
  2. Horizontal morphological opening (long kernel) to isolate staff lines
  3. Row-projection profile → peaks = Y of each line
  4. Group consecutive lines into StaffGroups of 5
  5. Return StaffGroup list with line Y positions + inter-line spacing
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StaffGroup:
    """
    Represents one 5-line staff system detected in the image.

    Attributes:
        line_ys:        Sorted list of 5 Y-coordinates (one per staff line),
                        from top line to bottom line.
        interline:      Average vertical distance between adjacent lines (pixels).
        top:            Y of the topmost line.
        bottom:         Y of the bottommost line.
    """
    line_ys: List[int] = field(default_factory=list)

    @property
    def interline(self) -> float:
        """Average spacing between adjacent staff lines."""
        if len(self.line_ys) < 2:
            return 0.0
        diffs = [self.line_ys[i + 1] - self.line_ys[i] for i in range(len(self.line_ys) - 1)]
        return sum(diffs) / len(diffs)

    @property
    def top(self) -> int:
        return self.line_ys[0] if self.line_ys else 0

    @property
    def bottom(self) -> int:
        return self.line_ys[-1] if self.line_ys else 0

    @property
    def center(self) -> float:
        """Vertical center of this staff group."""
        return (self.top + self.bottom) / 2.0


def detect_staff_lines(image_path: str, min_line_width_ratio: float = 0.3) -> List[StaffGroup]:
    """
    Detect staff line groups in a music sheet image.

    Args:
        image_path:           Path to the input image.
        min_line_width_ratio: Minimum width of detected lines relative to image width.
                              Lines shorter than this ratio are discarded (noise).
                              Default 0.3 means lines must span at least 30% of image width.

    Returns:
        List of StaffGroup objects, sorted top to bottom.
    """
    # --- Step 1: Load and binarize ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"[StaffDetector] Cannot read image: {image_path}")

    h, w = img.shape
    print(f"[StaffDetector] Image loaded: {w}×{h}")

    # Otsu binarization — invert so foreground (ink) = white
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Step 2: Morphological horizontal opening ---
    # A very wide, very short kernel isolates long horizontal structures (staff lines)
    # while destroying everything else (noteheads, stems, text).
    kernel_width = max(int(w * min_line_width_ratio), 30)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # --- Step 3: Row projection profile ---
    # Sum each row horizontally. Rows that are part of a staff line will have high sums.
    row_projection = np.sum(horizontal_lines, axis=1) / 255  # normalize to pixel count

    # A row is considered a "line row" if at least min_line_width_ratio of the image is white
    threshold = w * min_line_width_ratio * 0.5
    line_mask = row_projection > threshold

    # --- Step 4: Cluster consecutive True rows into single line Y values ---
    line_ys = _cluster_line_rows(line_mask)
    print(f"[StaffDetector] Raw lines detected: {len(line_ys)} → Y positions: {line_ys[:20]}...")

    if len(line_ys) < 5:
        print("[StaffDetector] ⚠️ Fewer than 5 lines found — falling back to heuristic mode")
        return _fallback_single_staff(h)

    # --- Step 5: Group lines into staffs of 5 ---
    staffs = _group_into_staffs(line_ys)
    print(f"[StaffDetector] ✅ Detected {len(staffs)} staff group(s)")
    for i, s in enumerate(staffs):
        print(f"  Staff {i + 1}: lines at Y={s.line_ys}, interline={s.interline:.1f}px")

    return staffs


def _cluster_line_rows(line_mask: np.ndarray, gap_tolerance: int = 3) -> List[int]:
    """
    Cluster consecutive True values in the row mask into single Y coordinates.

    When multiple consecutive rows are part of the same staff line (line has
    thickness > 1px), we take the median Y as the representative coordinate.

    Args:
        line_mask:      Boolean array, length = image height.
        gap_tolerance:  Max gap between rows still considered same line.

    Returns:
        Sorted list of representative Y coordinates.
    """
    line_ys = []
    in_run = False
    run_start = 0

    for y in range(len(line_mask)):
        if line_mask[y]:
            if not in_run:
                in_run = True
                run_start = y
        else:
            if in_run:
                # End of a run — compute center Y
                run_end = y - 1
                center = (run_start + run_end) // 2
                line_ys.append(center)
                in_run = False

    # Handle run that extends to the bottom of the image
    if in_run:
        center = (run_start + len(line_mask) - 1) // 2
        line_ys.append(center)

    # Merge line Y values that are too close together (within gap_tolerance)
    if not line_ys:
        return []

    merged = [line_ys[0]]
    for y in line_ys[1:]:
        if y - merged[-1] <= gap_tolerance:
            # Merge: take the average
            merged[-1] = (merged[-1] + y) // 2
        else:
            merged.append(y)

    return merged


def _group_into_staffs(line_ys: List[int]) -> List[StaffGroup]:
    """
    Group detected line Y-coordinates into StaffGroups of 5 lines each.

    Strategy:
      1. Compute all inter-line gaps.
      2. The median gap is the typical intra-staff spacing.
      3. Gaps significantly larger than median indicate staff-group boundaries.
      4. Split at those boundaries, then validate each group has exactly 5 lines.
    """
    if len(line_ys) < 5:
        return []

    # Compute gaps between consecutive lines
    gaps = [line_ys[i + 1] - line_ys[i] for i in range(len(line_ys) - 1)]
    median_gap = float(np.median(gaps))

    # A gap > 2.5× the median signals a new staff group
    split_threshold = median_gap * 2.5

    # Split into groups
    groups = []
    current_group = [line_ys[0]]

    for i in range(1, len(line_ys)):
        if line_ys[i] - line_ys[i - 1] > split_threshold:
            groups.append(current_group)
            current_group = [line_ys[i]]
        else:
            current_group.append(line_ys[i])

    groups.append(current_group)

    # Build StaffGroup objects — only accept groups with exactly 5 lines
    staffs = []
    for group in groups:
        if len(group) == 5:
            staffs.append(StaffGroup(line_ys=group))
        elif len(group) > 5:
            # If we got more than 5 (noise), take the 5 most evenly spaced
            # Simple approach: just take every Nth to get closest to 5
            print(f"[StaffDetector] ⚠️ Group with {len(group)} lines — trimming to 5")
            # Take first 5 for now (most common case is slight noise)
            staffs.append(StaffGroup(line_ys=group[:5]))
        else:
            print(f"[StaffDetector] ⚠️ Ignoring group with only {len(group)} lines: {group}")

    return staffs


def _fallback_single_staff(image_height: int) -> List[StaffGroup]:
    """
    Fallback when line detection fails: assume a single staff centered in the image.
    Uses standard music engraving proportions.
    """
    print("[StaffDetector] Using fallback staff estimation")
    center_y = image_height // 2
    # Standard staff: 4 spaces between 5 lines
    estimated_interline = image_height // 40  # rough estimate
    half_staff = estimated_interline * 2

    lines = [
        center_y - 2 * estimated_interline,
        center_y - estimated_interline,
        center_y,
        center_y + estimated_interline,
        center_y + 2 * estimated_interline,
    ]
    return [StaffGroup(line_ys=lines)]


def find_closest_staff(y_position: float, staffs: List[StaffGroup]) -> Optional[StaffGroup]:
    """
    Find which StaffGroup a given Y position belongs to.
    Returns the staff whose vertical center is closest to y_position.
    """
    if not staffs:
        return None
    return min(staffs, key=lambda s: abs(s.center - y_position))


def y_to_staff_position(y: float, staff: StaffGroup) -> float:
    """
    Convert a Y pixel coordinate to a staff position number.

    Returns a float where:
      0.0 = bottom line (line 1 in music notation, E4 in treble clef)
      1.0 = first space above bottom line (F4 in treble clef)
      2.0 = second line (G4 in treble clef)
      ...
      8.0 = top line (F5 in treble clef)

    Values can extend beyond 0-8 for ledger lines above/below the staff.
    Negative = below the staff, >8 = above the staff.
    """
    if staff.interline == 0:
        return 4.0  # fallback to middle of staff

    # Bottom line is the reference (position 0)
    bottom_y = staff.line_ys[-1]  # highest Y value = bottom of staff on screen

    # In image coordinates, Y increases downward, but pitch increases upward
    # Each interline = 2 staff positions (line + space)
    position_from_bottom = (bottom_y - y) / (staff.interline / 2.0)

    return position_from_bottom
