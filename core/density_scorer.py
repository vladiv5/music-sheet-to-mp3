"""
core/density_scorer.py
-----------------------
Sheet Music Density Scorer — Pure OpenCV Pre-Analysis

Evaluates how "dense" (complex) a sheet music image is BEFORE any ML inference.
Used for Smart Auto routing: simple pages → Custom YOLO (fast), dense pages → Oemer
or SAHI-enhanced YOLO.

The score is a float in [0.0, 1.0] combining 3 independent OpenCV metrics:
  1. Ink Density     — ratio of black pixels to total pixels
  2. Vertical Complexity — transitions on vertical columns (stacked notes)
  3. Staff Utilization   — how packed each staff region is

This module adds <100ms to the pipeline (pure NumPy/OpenCV, no ML).
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Try to import StaffGroup for type hints, but don't fail if unavailable
try:
    from core.staff_detector import StaffGroup, detect_staff_lines
except ImportError:
    StaffGroup = None
    detect_staff_lines = None


@dataclass
class DensityReport:
    """Detailed breakdown of the density analysis."""
    ink_density: float          # 0.0–1.0: ratio of dark pixels
    vertical_complexity: float  # 0.0–1.0: vertical transition density
    staff_utilization: float    # 0.0–1.0: how packed the staff regions are
    overall_score: float        # 0.0–1.0: weighted combination
    recommendation: str         # 'yolo', 'sahi', or 'oemer'
    num_staffs: int             # number of detected staff groups

    @property
    def label(self) -> str:
        """Human-readable complexity label."""
        if self.overall_score < 0.20:
            return "🟢 Very Simple"
        elif self.overall_score < 0.35:
            return "🟡 Moderate"
        elif self.overall_score < 0.55:
            return "🟠 Dense"
        else:
            return "🔴 Very Dense"


def compute_density_score(
    image_path: str,
    density_threshold: float = 0.35,
    weights: Tuple[float, float, float] = (0.30, 0.35, 0.35),
) -> DensityReport:
    """
    Compute a density score for a sheet music image.

    Args:
        image_path:        Path to the input image.
        density_threshold: Score above which SAHI/Oemer is recommended.
        weights:           (w_ink, w_vertical, w_staff) weights for final score.

    Returns:
        DensityReport with individual metrics and recommendation.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"[DensityScorer] Cannot read image: {image_path}")

    h, w = gray.shape
    print(f"[DensityScorer] Analyzing image: {w}×{h}")

    # Binarize (ink = white on black background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Metric 1: Ink Density ---
    ink_density = _compute_ink_density(binary)

    # --- Metric 2: Vertical Complexity ---
    vertical_complexity = _compute_vertical_complexity(binary, h, w)

    # --- Metric 3: Staff Utilization ---
    staff_utilization, num_staffs = _compute_staff_utilization(binary, image_path, h, w)

    # --- Weighted combination ---
    w_ink, w_vert, w_staff = weights
    overall = (w_ink * ink_density +
               w_vert * vertical_complexity +
               w_staff * staff_utilization)

    # Clamp to [0, 1]
    overall = max(0.0, min(1.0, overall))

    # Recommendation
    if overall < density_threshold:
        recommendation = 'yolo'
    elif overall < density_threshold + 0.20:
        recommendation = 'sahi'
    else:
        recommendation = 'oemer'

    report = DensityReport(
        ink_density=round(ink_density, 3),
        vertical_complexity=round(vertical_complexity, 3),
        staff_utilization=round(staff_utilization, 3),
        overall_score=round(overall, 3),
        recommendation=recommendation,
        num_staffs=num_staffs,
    )

    print(f"[DensityScorer] Results: ink={report.ink_density:.3f}, "
          f"vert={report.vertical_complexity:.3f}, "
          f"staff={report.staff_utilization:.3f} → "
          f"overall={report.overall_score:.3f} ({report.label}) → {report.recommendation}")

    return report


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def _compute_ink_density(binary: np.ndarray) -> float:
    """
    Ratio of ink (dark) pixels to total pixels.
    
    A blank page ≈ 0.0, a heavily annotated page ≈ 0.15–0.25.
    Normalized so that 0.20 ink → ~1.0 output.
    """
    total = binary.shape[0] * binary.shape[1]
    ink_pixels = np.count_nonzero(binary)
    raw_ratio = ink_pixels / total

    # Normalize: typical range is 0.03 (very sparse) to 0.20 (very dense)
    # Map [0.03, 0.20] → [0.0, 1.0]
    normalized = (raw_ratio - 0.03) / (0.20 - 0.03)
    return max(0.0, min(1.0, normalized))


def _compute_vertical_complexity(binary: np.ndarray, h: int, w: int) -> float:
    """
    Count vertical transitions (white→black→white) per column.
    
    A single melody line has ~2 transitions per column (enter note, exit note).
    Dense chords have 6-10+ transitions per column.
    """
    # Sample every 4th column for speed
    step = max(1, w // 300)
    sampled_cols = binary[:, ::step]

    # Count transitions per column: diff between adjacent rows
    diffs = np.abs(np.diff(sampled_cols.astype(np.int16), axis=0))
    transitions_per_col = np.sum(diffs > 127, axis=0)

    # Average transitions per column
    avg_transitions = np.mean(transitions_per_col)

    # Normalize: range [2, 14] → [0.0, 1.0]
    # 2 transitions = simple melody, 14+ = extreme polyphony
    normalized = (avg_transitions - 2.0) / (14.0 - 2.0)
    return max(0.0, min(1.0, normalized))


def _compute_staff_utilization(
    binary: np.ndarray,
    image_path: str,
    h: int, w: int
) -> Tuple[float, int]:
    """
    Measure how packed each staff region is with ink.
    
    Strategy:
      1. Detect staff groups (reuse staff_detector)
      2. For each staff, compute ink density in the staff region
      3. Average across staffs
    """
    num_staffs = 0

    try:
        if detect_staff_lines is not None:
            staffs = detect_staff_lines(image_path)
            num_staffs = len(staffs)
        else:
            staffs = []
    except Exception:
        staffs = []

    if not staffs:
        # Fallback: use overall ink density as proxy
        total = h * w
        ink = np.count_nonzero(binary)
        return min(1.0, (ink / total) / 0.15), 0

    utilizations = []
    for staff in staffs:
        # Define the staff region: from 1 interline above top line to 1 below bottom
        margin = int(staff.interline * 1.5) if staff.interline > 0 else 20
        y_top = max(0, staff.top - margin)
        y_bot = min(h, staff.bottom + margin)

        if y_bot <= y_top:
            continue

        staff_roi = binary[y_top:y_bot, :]
        roi_pixels = staff_roi.shape[0] * staff_roi.shape[1]
        if roi_pixels == 0:
            continue

        ink_in_staff = np.count_nonzero(staff_roi)
        utilizations.append(ink_in_staff / roi_pixels)

    if not utilizations:
        return 0.0, num_staffs

    avg_util = np.mean(utilizations)

    # Normalize: range [0.05, 0.25] → [0.0, 1.0]
    normalized = (avg_util - 0.05) / (0.25 - 0.05)
    return max(0.0, min(1.0, normalized)), num_staffs
