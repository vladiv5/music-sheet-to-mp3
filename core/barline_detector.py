import cv2
import numpy as np
from typing import List, Dict, Any
from core.staff_detector import StaffGroup

def detect_barlines(image_path: str, staffs: List[StaffGroup], detections: List[Any] = None) -> Dict[int, List[int]]:
    """
    Detects vertical barlines in the score using OpenCV morphological operations.
    Returns:
        A dictionary mapping id(staff) -> sorted list of X coordinates for barlines.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"[BarlineDetector] Image not found: {image_path}")

    # Binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # We need to find vertical lines.
    # The height of a simple staff is 4 * interline.
    # Let's assume interline is roughly the same across staffs.
    interline = 10.0
    if staffs and len(staffs[0].line_ys) >= 2:
        interline = staffs[0].interline

    # Create a vertical structure element.
    # A barline must be at least 3.5 interlines tall.
    kernel_h = max(int(interline * 3.5), 15)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # We might have noise (e.g. note stems). Note stems are also vertical lines.
    # But stems are rarely taller than 3.5 * interline unless they are beamed 
    # chords. True barlines usually run exactly from the top line of a staff 
    # to the bottom line (4 interlines), or across the whole system.
    # Let's find contours and filter them.
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    barline_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # A barline is a thin vertical line
        if w < interline * 1.5 and h >= interline * 3.5:
            barline_candidates.append((x + w/2.0, y, y + h))

    # FILTER 1: Remove Stems!
    # A vertical line (stem) belonging to a note will share its X coordinate with the notehead.
    # Noteheads are approx 1.0 - 1.5 interlines wide. Stems are attached to the side.
    # We discard any vertical line whose X-center is within 1.2 interlines of a notehead.
    note_xs = []
    if detections:
        for d in detections:
            if 'notehead' in d.class_name or 'flag' in d.class_name:
                note_xs.append(d.x_center)
                
    valid_candidates = []
    for bx, by1, by2 in barline_candidates:
        h_line = by2 - by1
        is_stem = False
        
        # If the vertical line is extremely tall (e.g. spanning multiple staves in a grand staff),
        # it is definitely a barline, not a stem. Stems rarely exceed 4.5 interlines.
        if h_line < interline * 6.0:
            for nx in note_xs:
                # Tightened threshold: a stem is usually 0.5 - 0.6 interlines away from the notehead center.
                if abs(bx - nx) < interline * 0.85:
                    is_stem = True
                    break
                    
        if not is_stem:
            valid_candidates.append((bx, by1, by2))

    # FILTER 2: Assign valid barlines to staffs
    staff_barlines = {id(staff): [] for staff in staffs}
    
    for staff in staffs:
        s_top = staff.top
        s_bottom = staff.bottom
        s_center = staff.center
        
        # A bit of tolerance
        tolerance = interline * 1.5
        xs = []
        
        for bx, by1, by2 in valid_candidates:
            # Check if the vertical line overlaps the staff
            # The line should start near or above s_top, and end near or below s_bottom
            if by1 <= s_top + tolerance and by2 >= s_bottom - tolerance:
                xs.append(int(bx))
        
        # Sort and remove duplicates (sometimes thick bars get broken into multiple contours)
        xs.sort()
        filtered_xs = []
        for x in xs:
            if not filtered_xs or abs(x - filtered_xs[-1]) > interline * 2:
                filtered_xs.append(x)
                
        staff_barlines[id(staff)] = filtered_xs
        print(f"[BarlineDetector] Found {len(filtered_xs)} barlines for staff at Y={staff.center:.1f}")

    return staff_barlines
