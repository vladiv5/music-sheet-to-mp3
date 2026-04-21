"""
core/primitive_assembler.py
---------------------------
Fuses YOLO primitive detections (e.g. noteheadBlack, flag8thUp) with 
classical CV (beam counts) into coherent musical objects.
"""
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np
from dataclasses import dataclass

# Allow lazy import or duck typing for Detection
# Assuming detection objects have: class_name, x_center, y_center, width, height, confidence

# Constants for duration
DURATIONS = {
    'whole': 4.0,
    'half': 2.0,
    'quarter': 1.0,
    'eighth': 0.5,
    '16th': 0.25,
    '32nd': 0.125
}

def find_closest_primitive(target, candidates, max_dist_x, max_dist_y):
    """Find the closest bounding box among candidates to the target."""
    best = None
    min_dist = float('inf')
    for cand in candidates:
        dx = abs(target.x_center - cand.x_center)
        dy = abs(target.y_center - cand.y_center)
        if dx < max_dist_x and dy < max_dist_y:
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                best = cand
    return best


def assemble_primitives(detections: List[any], gray_image: np.ndarray, interline: float, enable_beam_correction: bool = True) -> List[any]:
    """
    Combines YOLO primitive symbols into full representations.
    For now, it simply finds noteheads and infers their rhythms:
    1. Check for flags.
    2. Fallback to beam_detector (OpenCV) to count lines.
    
    Returns a new list of detections where noteheads are replaced by 'Quarter-note', etc.,
    so the existing inference logic can process them.
    """
    from core.beam_detector import detect_beams_for_note

    assembled = []
    
    # Separate notes from other primitives
    notes = [d for d in detections if d.class_name in ['noteheadBlack', 'noteheadEmpty']]
    flags = [d for d in detections if d.class_name.startswith('flag')]
    beams = [d for d in detections if d.class_name == 'beam']
    dots = [d for d in detections if d.class_name == 'augmentationDot']
    accidentals = [d for d in detections if d.class_name.startswith('accidental')]
    others = [d for d in detections if d not in notes and d not in flags and d not in beams]

    for n in notes:
        is_empty = (n.class_name == 'noteheadEmpty')
        
        # 1. Determine rhythm
        duration_type = 'quarter'
        
        if is_empty:
            # We must check if it's a half note or whole note. (usually whole has no stem)
            # We assume it's a half note for now. Could check stem existence.
            duration_type = 'half'
        else:
            # 1. Check for YOLO beam overlaps
            note_beams = 0
            for b in beams:
                b_left = b.x_center - b.width / 2
                b_right = b.x_center + b.width / 2
                # If notehead's X is under the beam's X-range (with small tolerance)
                if b_left - (interline) <= n.x_center <= b_right + (interline):
                    # And beam is vertically close to the notehead (max 8 interlines away)
                    if abs(b.y_center - n.y_center) < interline * 8:
                        note_beams += 1
            
            if note_beams > 0:
                if note_beams == 1:
                    duration_type = 'eighth'
                elif note_beams >= 2:
                    duration_type = '16th'
            else:
                # 2. Check flags from YOLO
                closest_flag = find_closest_primitive(n, flags, max_dist_x=max(interline*2.5, 30), max_dist_y=max(interline*4, 60))
                if closest_flag:
                    if '8th' in closest_flag.class_name:
                        duration_type = 'eighth'
                    elif '16th' in closest_flag.class_name:
                        duration_type = '16th'
                else:
                    # 3. Fallback to OpenCV beam detection if enabled
                    if enable_beam_correction:
                        expanded_h = max(n.height, interline * 3.5)
                        beam_count = detect_beams_for_note(
                            x_center=n.x_center,
                            y_center=n.y_center, 
                            width=n.width,
                            height=expanded_h * 2,
                            class_name='Quarter-note',  
                            gray_image=gray_image,
                            interline=interline,
                            notehead_y=n.y_center
                        )
                        
                        if beam_count == 1:
                            duration_type = 'eighth'
                        elif beam_count == 2:
                            duration_type = '16th'
                        elif beam_count == 3:
                            duration_type = '32nd'
                        else:
                            duration_type = 'quarter'
                    else:
                        duration_type = 'quarter'
                    
        # 2. Check for augmentation dot
        closest_dot = find_closest_primitive(n, dots, max_dist_x=max(interline*2, 20), max_dist_y=n.height)
        new_class_name = ""
        
        if duration_type == 'whole':
            new_class_name = 'Full-note'
        elif duration_type == 'half':
            new_class_name = 'Half-dot' if closest_dot else 'Half-note'
        elif duration_type == 'quarter':
            new_class_name = 'Quarter-dot' if closest_dot else 'Quarter-note'
        elif duration_type == 'eighth':
            new_class_name = 'Eighth-note'
        elif duration_type == '16th':
            new_class_name = 'Sixteenth-note'
        else:
            new_class_name = 'Quarter-note' # Default
            
        # Create an updated detection object
        # We mutate or copy the detection to act like an old-style note
        n.class_name = new_class_name
        assembled.append(n)
        
    # Translate rests and accidentals back to old format so we can reuse logic,
    # OR we handle the mapping in primitive_yolo_inference.
    
    # Actually, it's cleaner to return the primitives with generic class names
    # mapped safely so `primitive_yolo_inference` can use them directly.
    return assembled + others
