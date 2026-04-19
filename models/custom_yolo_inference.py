"""
models/custom_yolo_inference.py
--------------------------------
Custom YOLOv8 Inference Engine for OMR (Optical Music Recognition)

This module replaces the previous mock with real inference using the custom-trained
YOLOv8s model (15 classes, 1280px). It combines YOLO object detection with classical
CV staff-line detection to produce accurate MusicXML output.

Pipeline:
  1. YOLO detects musical symbols + bounding boxes
  2. OpenCV detects staff lines (via core.staff_detector)
  3. Combinator maps each note's Y position to the staff grid → exact pitch
  4. Generator builds a valid MusicXML document

Architecture note:
  The function signature of generation_workflow_custom_yolo() is intentionally
  identical to generation_workflow_oemer() so that app.py can swap engines
  transparently via a single dispatcher block.
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement, ElementTree, tostring
from xml.dom import minidom

# Ensure the project root is on sys.path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.staff_detector import detect_staff_lines, find_closest_staff, y_to_staff_position, StaffGroup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8s_15clase_1280px.pt")

# The 15 classes from big_dataset/data.yaml (index order matters!)
CLASS_NAMES = [
    'Bass-clef', 'Bemol', 'Diez', 'Eighth-note', 'Eighth-rest',
    'Full-note', 'Full-rest', 'Half-dot', 'Half-note', 'Half-rest',
    'Quarter-dot', 'Quarter-note', 'Quarter-rest', 'Sixteenth-note',
    'Treble-clef'
]

# Mapping: class name → note duration type (MusicXML duration types)
NOTE_CLASSES = {
    'Full-note':      ('whole',    4),
    'Half-note':      ('half',     2),
    'Half-dot':       ('half',     3),  # dotted half = 3 beats
    'Quarter-note':   ('quarter',  1),
    'Quarter-dot':    ('quarter',  1.5),  # dotted quarter
    'Eighth-note':    ('eighth',   0.5),
    'Sixteenth-note': ('16th',     0.25),
}

REST_CLASSES = {
    'Full-rest':    ('whole',    4),
    'Half-rest':    ('half',     2),
    'Quarter-rest': ('quarter',  1),
    'Eighth-rest':  ('eighth',   0.5),
}

CLEF_CLASSES = {'Treble-clef', 'Bass-clef'}
ACCIDENTAL_CLASSES = {'Bemol': 'flat', 'Diez': 'sharp'}

# Treble clef pitch mapping: staff_position → (step, octave)
# Position 0 = bottom line = E4, position 1 = first space = F4, etc.
TREBLE_PITCH_MAP = {
    -6: ('F', 3),
    -5: ('G', 3),
    -4: ('A', 3),
    -3: ('B', 3),
    -2: ('C', 4),    # Middle C (1st ledger line below)
    -1: ('D', 4),    # space below bottom line
    0:  ('E', 4),    # bottom line
    1:  ('F', 4),
    2:  ('G', 4),
    3:  ('A', 4),
    4:  ('B', 4),    # middle line
    5:  ('C', 5),
    6:  ('D', 5),
    7:  ('E', 5),
    8:  ('F', 5),    # top line
    9:  ('G', 5),
    10: ('A', 5),
    11: ('B', 5),
    12: ('C', 6),    # ledger line above
    13: ('D', 6),
    14: ('E', 6),
}

# Bass clef pitch mapping: position 0 = bottom line = G2
BASS_PITCH_MAP = {
    -6: ('A', 1),
    -5: ('B', 1),
    -4: ('C', 2),
    -3: ('D', 2),
    -2: ('E', 2),
    -1: ('F', 2),
    0:  ('G', 2),    # bottom line
    1:  ('A', 2),
    2:  ('B', 2),
    3:  ('C', 3),
    4:  ('D', 3),    # middle line
    5:  ('E', 3),
    6:  ('F', 3),
    7:  ('G', 3),
    8:  ('A', 3),    # top line
    9:  ('B', 3),
    10: ('C', 4),    # Middle C (1st ledger line above)
    11: ('D', 4),
    12: ('E', 4),
    13: ('F', 4),
    14: ('G', 4),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single YOLO detection result."""
    class_name: str
    x_center: float     # absolute pixel coordinate
    y_center: float
    width: float
    height: float
    confidence: float
    class_id: int


@dataclass
class MusicalEvent:
    """A musical event ready for MusicXML generation."""
    event_type: str      # 'note', 'rest'
    duration_type: str   # 'quarter', 'half', 'whole', 'eighth', '16th'
    duration_beats: float
    step: Optional[str] = None    # 'C', 'D', etc.
    octave: Optional[int] = None
    alter: Optional[int] = None   # -1 = flat, +1 = sharp
    x_position: float = 0.0       # for sorting left-to-right


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_model = None


def load_model():
    """Load the YOLO model once and cache it globally."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        print(f"[CustomYOLO] 📦 Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"[CustomYOLO] ❌ Model file not found: {MODEL_PATH}\n"
                f"Expected: yolov8s_15clase_1280px.pt in the models/ directory."
            )
        _model = YOLO(MODEL_PATH)
        print("[CustomYOLO] ✅ Model loaded successfully")
    return _model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def run_detection(image_path: str, conf: float = 0.25) -> List[Detection]:
    """
    Run YOLO inference on the image.

    Args:
        image_path: Path to the input image.
        conf:       Confidence threshold (0.0–1.0).

    Returns:
        List of Detection objects, sorted left-to-right by x_center.
    """
    model = load_model()

    print(f"[CustomYOLO] 🔍 Running inference (conf={conf}, imgsz=1280)...")
    results = model.predict(
        source=image_path,
        imgsz=1280,
        conf=conf,
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            x_c, y_c, w, h = boxes.xywh[i].tolist()
            conf_val = boxes.conf[i].item()

            detections.append(Detection(
                class_name=cls_name,
                x_center=x_c,
                y_center=y_c,
                width=w,
                height=h,
                confidence=conf_val,
                class_id=cls_id,
            ))

    # Sort left-to-right (temporal order)
    detections.sort(key=lambda d: d.x_center)

    print(f"[CustomYOLO] 📋 Detected {len(detections)} symbols:")
    class_counts = {}
    for d in detections:
        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
    for name, count in sorted(class_counts.items()):
        print(f"    {name}: {count}")

    return detections


# ---------------------------------------------------------------------------
# Pitch mapping
# ---------------------------------------------------------------------------

def _get_pitch_map(clef_type: str) -> Dict[int, Tuple[str, int]]:
    """Return the appropriate pitch map for the clef type."""
    if clef_type == 'Bass-clef':
        return BASS_PITCH_MAP
    return TREBLE_PITCH_MAP


def find_notehead_y(detection: Detection, gray_image: np.ndarray) -> float:
    """
    Find the exact Y coordinate of the notehead within the bounding box.
    Uses contour detection to isolate the notehead blob (the "circle")
    and calculates its center based on its extremities, ignoring the stem.
    """
    if detection.class_name == 'Full-note':
        return detection.y_center

    h, w = gray_image.shape
    x1 = max(0, int(detection.x_center - detection.width / 2))
    x2 = min(w, int(detection.x_center + detection.width / 2))
    y1 = max(0, int(detection.y_center - detection.height / 2))
    y2 = min(h, int(detection.y_center + detection.height / 2))

    if y2 <= y1 or x2 <= x1:
        return detection.y_center

    box_roi = gray_image[y1:y2, x1:x2]
    
    # Binarize the image (ink = 255, background = 0)
    _, binary = cv2.threshold(box_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. Fill hollow notes (like Half-notes) so they become solid blobs
    # We do this by finding contours and drawing them filled
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_binary = np.zeros_like(binary)
    cv2.drawContours(filled_binary, contours, -1, 255, thickness=cv2.FILLED)
    
    # 2. Erase the thin stem and thin staff lines using morphological opening
    # A 3x3 or 4x4 kernel will delete lines that are 1-2 pixels thick, leaving the fat notehead
    kernel_size = max(3, int(min(detection.width, detection.height) * 0.15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_blob = cv2.morphologyEx(filled_binary, cv2.MORPH_OPEN, kernel)
    
    # 3. Find the isolated notehead contour
    final_contours, _ = cv2.findContours(cleaned_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not final_contours:
        # Fallback if opening erased everything (e.g., box too small)
        final_contours = contours
        if not final_contours:
            return detection.y_center

    # 4. Pick the largest blob (which must be the notehead)
    best_contour = max(final_contours, key=cv2.contourArea)
    
    # 5. Get the 4 extreme coordinates of the blob (sus, jos, stanga, dreapta)
    # cv2.boundingRect naturally gives us exactly the bounding limits of this specific blob!
    cx, cy, cw, ch = cv2.boundingRect(best_contour)
    
    # The center of this blob is exactly what we need
    notehead_center_y = cy + (ch / 2.0)
    
    return y1 + notehead_center_y


def map_note_pitch(
    note_y: float,
    staffs: List[StaffGroup],
    current_clef: str = 'Treble-clef'
) -> Tuple[str, int]:
    """
    Map a detected note's Y position to a musical pitch (step + octave).
    """
    staff = find_closest_staff(note_y, staffs)
    if staff is None:
        # Fallback: return middle C
        return ('C', 4)

    # Get raw staff position (float)
    raw_position = y_to_staff_position(note_y, staff)

    # Round to nearest half-position (line or space)
    position = round(raw_position)

    # Clamp to the pitch map range
    pitch_map = _get_pitch_map(current_clef)
    min_pos = min(pitch_map.keys())
    max_pos = max(pitch_map.keys())
    position = max(min_pos, min(max_pos, position))

    step, octave = pitch_map[position]
    return (step, octave)


# ---------------------------------------------------------------------------
# MusicXML generation
# ---------------------------------------------------------------------------

def _find_accidental_for_note(
    note_det: Detection,
    all_detections: List[Detection],
    max_distance: float = 50.0
) -> Optional[int]:
    """
    Check if there is an accidental (Bemol/Diez) immediately to the left of a note.
    Returns -1 for flat, +1 for sharp, None for natural.
    """
    for det in all_detections:
        if det.class_name in ACCIDENTAL_CLASSES:
            # Must be to the left and vertically close
            x_dist = note_det.x_center - det.x_center
            y_dist = abs(note_det.y_center - det.y_center)
            if 0 < x_dist < max_distance and y_dist < note_det.height:
                return -1 if det.class_name == 'Bemol' else 1
    return None


def detections_to_events(
    detections: List[Detection],
    staffs: List[StaffGroup],
    gray_image: np.ndarray
) -> List[MusicalEvent]:
    """
    Convert raw YOLO detections into a sequence of MusicalEvents.
    Groups detections by staff (top-to-bottom) and processes them
    left-to-right to ensure correct musical reading order.
    """
    events = []
    
    # 1. Group detections by their assigned staff
    # Create a list of empty lists, one for each staff
    staff_detections = {id(staff): [] for staff in staffs}
    orphaned_detections = []
    
    for det in detections:
        # For clefs, notes, etc, the standard y_center works well enough for staff assignment
        staff = find_closest_staff(det.y_center, staffs)
        if staff is not None:
            staff_detections[id(staff)].append(det)
        else:
            orphaned_detections.append(det)

    if orphaned_detections and not staffs:
        # Fallback if no staffs were detected at all
        staff_detections['fallback'] = orphaned_detections
    
    # 2. Process each staff from top to bottom
    for staff in staffs:
        group = staff_detections.get(id(staff), [])
        
        # Sort left-to-right within this specific staff
        group.sort(key=lambda d: d.x_center)
        
        # Default clef at the start of a staff line could be inherited,
        # but typically it's Treble unless a Bass clef is explicitly defined.
        current_clef = 'Treble-clef'

        for det in group:
            # Track clef changes (affects pitch interpretation on THIS staff)
            if det.class_name in CLEF_CLASSES:
                current_clef = det.class_name
                print(f"[CustomYOLO] 🎼 Clef detected: {current_clef} at x={det.x_center:.0f} (Staff top Y={staff.top})")
                continue

            # Skip accidentals — they will be attached to the next note
            if det.class_name in ACCIDENTAL_CLASSES:
                continue

            # Process notes
            if det.class_name in NOTE_CLASSES:
                duration_type, duration_beats = NOTE_CLASSES[det.class_name]
                
                # Use exact notehead Y to compute pitch
                note_y = find_notehead_y(det, gray_image)
                # Map pitch based on the CURRENT staff and its CURRENT clef
                step, octave = map_note_pitch(note_y, [staff], current_clef)
                
                alter = _find_accidental_for_note(det, group)

                events.append(MusicalEvent(
                    event_type='note',
                    duration_type=duration_type,
                    duration_beats=duration_beats,
                    step=step,
                    octave=octave,
                    alter=alter,
                    x_position=det.x_center,
                ))

            # Process rests
            elif det.class_name in REST_CLASSES:
                duration_type, duration_beats = REST_CLASSES[det.class_name]
                events.append(MusicalEvent(
                    event_type='rest',
                    duration_type=duration_type,
                    duration_beats=duration_beats,
                    x_position=det.x_center,
                ))

    print(f"[CustomYOLO] 🎵 Generated {len(events)} musical events "
          f"({sum(1 for e in events if e.event_type == 'note')} notes, "
          f"{sum(1 for e in events if e.event_type == 'rest')} rests)")

    return events


def _split_into_measures(events: List[MusicalEvent], beats_per_measure: int = 4) -> List[List[MusicalEvent]]:
    """
    Split a flat list of events into measures based on beat count.
    Uses a simple greedy approach: fill each measure up to beats_per_measure.
    """
    if not events:
        return [[]]

    measures = []
    current_measure = []
    current_beats = 0.0

    for event in events:
        if current_beats + event.duration_beats > beats_per_measure + 0.01:
            # Start a new measure
            if current_measure:
                measures.append(current_measure)
            current_measure = [event]
            current_beats = event.duration_beats
        else:
            current_measure.append(event)
            current_beats += event.duration_beats

    if current_measure:
        measures.append(current_measure)

    return measures


def events_to_musicxml(events: List[MusicalEvent], title: str = "Custom YOLO Detection") -> str:
    """
    Convert a list of MusicalEvents into a MusicXML string.
    """
    # Divisions: we use 4 divisions per quarter note to handle 16ths cleanly
    divisions = 4

    # Duration mapping: beats → MusicXML duration units (relative to divisions)
    def beats_to_duration(beats: float) -> int:
        return int(beats * divisions)

    # Build the XML tree
    score = Element('score-partwise', version='4.0')

    # Work title
    work = SubElement(score, 'work')
    SubElement(work, 'work-title').text = title

    # Identification
    identification = SubElement(score, 'identification')
    encoding = SubElement(identification, 'encoding')
    SubElement(encoding, 'software').text = 'Music Sheet to MP3 — Custom YOLO Engine'

    # Part list
    part_list = SubElement(score, 'part-list')
    score_part = SubElement(part_list, 'score-part', id='P1')
    SubElement(score_part, 'part-name').text = 'Piano'

    # Part content
    part = SubElement(score, 'part', id='P1')

    # Split events into measures
    measures = _split_into_measures(events)

    for m_idx, measure_events in enumerate(measures):
        measure = SubElement(part, 'measure', number=str(m_idx + 1))

        # First measure gets attributes (key, time, clef)
        if m_idx == 0:
            attributes = SubElement(measure, 'attributes')
            SubElement(attributes, 'divisions').text = str(divisions)
            key = SubElement(attributes, 'key')
            SubElement(key, 'fifths').text = '0'
            time_elem = SubElement(attributes, 'time')
            SubElement(time_elem, 'beats').text = '4'
            SubElement(time_elem, 'beat-type').text = '4'
            clef = SubElement(attributes, 'clef')
            SubElement(clef, 'sign').text = 'G'
            SubElement(clef, 'line').text = '2'

        for event in measure_events:
            note_elem = SubElement(measure, 'note')

            if event.event_type == 'rest':
                SubElement(note_elem, 'rest')
            else:
                pitch = SubElement(note_elem, 'pitch')
                SubElement(pitch, 'step').text = event.step
                if event.alter is not None:
                    SubElement(pitch, 'alter').text = str(event.alter)
                SubElement(pitch, 'octave').text = str(event.octave)

            SubElement(note_elem, 'duration').text = str(beats_to_duration(event.duration_beats))
            SubElement(note_elem, 'type').text = event.duration_type

            # Add dot for dotted notes
            if event.duration_type in ('half', 'quarter') and event.duration_beats in (3, 1.5):
                SubElement(note_elem, 'dot')

            # Add accidental element for display
            if event.alter is not None:
                acc_elem = SubElement(note_elem, 'accidental')
                acc_elem.text = 'flat' if event.alter == -1 else 'sharp'

    # If no events at all, add one empty measure with a whole rest
    if not events:
        measure = SubElement(part, 'measure', number='1')
        attributes = SubElement(measure, 'attributes')
        SubElement(attributes, 'divisions').text = str(divisions)
        key = SubElement(attributes, 'key')
        SubElement(key, 'fifths').text = '0'
        time_elem = SubElement(attributes, 'time')
        SubElement(time_elem, 'beats').text = '4'
        SubElement(time_elem, 'beat-type').text = '4'
        clef = SubElement(attributes, 'clef')
        SubElement(clef, 'sign').text = 'G'
        SubElement(clef, 'line').text = '2'
        note_elem = SubElement(measure, 'note')
        SubElement(note_elem, 'rest')
        SubElement(note_elem, 'duration').text = str(divisions * 4)
        SubElement(note_elem, 'type').text = 'whole'

    # Pretty-print
    rough_string = tostring(score, encoding='unicode')
    xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doctype = ('<!DOCTYPE score-partwise PUBLIC\n'
               '  "-//Recordare//DTD MusicXML 4.0 Partwise//EN"\n'
               '  "http://www.musicxml.org/dtds/partwise.dtd">\n')

    try:
        parsed = minidom.parseString(rough_string)
        pretty_body = parsed.toprettyxml(indent="  ", encoding=None)
        # Remove the xml declaration that minidom adds (we add our own)
        lines = pretty_body.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        pretty_body = '\n'.join(lines)
    except Exception:
        pretty_body = rough_string

    return xml_header + doctype + pretty_body


# ---------------------------------------------------------------------------
# Diagnostic image
# ---------------------------------------------------------------------------

def save_diagnostic_image(
    image_path: str,
    detections: List[Detection],
    staffs: List[StaffGroup],
    output_path: str
) -> str:
    """
    Save an annotated image showing:
      - Bounding boxes with class labels and confidence
      - Staff lines overlaid in blue
      - Color-coded boxes by category (notes=green, rests=orange, clefs=purple)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[CustomYOLO] ⚠️ Cannot read image for diagnostics: {image_path}")
        return output_path

    h, w = img.shape[:2]

    # Draw staff lines
    for staff in staffs:
        for y in staff.line_ys:
            cv2.line(img, (0, y), (w, y), (255, 100, 0), 1, cv2.LINE_AA)

    # Color scheme by category
    def get_color(class_name: str) -> Tuple[int, int, int]:
        if class_name in NOTE_CLASSES:
            return (0, 200, 0)       # green for notes
        elif class_name in REST_CLASSES:
            return (0, 140, 255)     # orange for rests
        elif class_name in CLEF_CLASSES:
            return (200, 0, 200)     # purple for clefs
        elif class_name in ACCIDENTAL_CLASSES:
            return (0, 255, 255)     # yellow for accidentals
        return (128, 128, 128)       # gray for unknown

    # Draw detections
    for det in detections:
        color = get_color(det.class_name)
        x1 = int(det.x_center - det.width / 2)
        y1 = int(det.y_center - det.height / 2)
        x2 = int(det.x_center + det.width / 2)
        y2 = int(det.y_center + det.height / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(output_path, img)
    print(f"[CustomYOLO] 🖼️ Diagnostic image saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main workflow (public API — same signature as Oemer baseline)
# ---------------------------------------------------------------------------

def generation_workflow_custom_yolo(
    image_path: str,
    output_dir: str = "./",
    conf: float = 0.25
) -> str:
    """
    Full custom YOLO inference pipeline: detect → map pitches → generate MusicXML.

    This function has the same signature as generation_workflow_oemer() (plus an
    optional `conf` parameter) so that app.py can swap engines transparently.

    Args:
        image_path: Path to the input sheet music image.
        output_dir: Directory for output files.
        conf:       YOLO confidence threshold (0.0–1.0).

    Returns:
        Absolute path to the generated .musicxml file.
    """
    print(f"[CustomYOLO] 🟢 Inference started — input: {image_path}")
    print(f"[CustomYOLO] 📂 Output: {output_dir}, Confidence: {conf}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Detect staff lines (classical CV) ---
    print("[CustomYOLO] 📏 Detecting staff lines...")
    staffs = detect_staff_lines(image_path)

    # Load grayscale image once for notehead exact extraction
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f"[CustomYOLO] Cannot read image: {image_path}")
    
    # --- Step 2: Run YOLO detection ---
    detections = run_detection(image_path, conf=conf)

    if not detections:
        print("[CustomYOLO] ⚠️ No symbols detected! Generating empty MusicXML.")

    # --- Step 3: Convert detections to musical events ---
    events = detections_to_events(detections, staffs, gray_image)

    # --- Step 4: Generate MusicXML ---
    xml_content = events_to_musicxml(events, title="Custom YOLO OMR Detection")
    output_xml_path = os.path.join(output_dir, "custom_yolo_output.musicxml")

    try:
        with open(output_xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
    except OSError as e:
        raise RuntimeError(
            f"[CustomYOLO] ❌ Could not write MusicXML to {output_xml_path}: {e}"
        ) from e

    print(f"[CustomYOLO] ✅ MusicXML written: {output_xml_path}")

    # --- Step 5: Save diagnostic image ---
    diag_path = os.path.join(output_dir, "yolo_detections.png")
    save_diagnostic_image(image_path, detections, staffs, diag_path)

    return output_xml_path
