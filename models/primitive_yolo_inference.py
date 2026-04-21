"""
models/primitive_yolo_inference.py
--------------------------------
Custom YOLOv8 Inference Engine for Primitive-Based OMR Model (35 classes).

Unlike the 15-class model which detects full notes (e.g. "Eighth-note"),
this engine uses a primitive-based model (e.g. "noteheadBlack", "flag8thUp")
and assembles them geometrically using core/primitive_assembler.py. 
We maintain identical signatures for app.py compatibility.
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
from core.primitive_assembler import assemble_primitives
from core.barline_detector import detect_barlines

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use the new primitive model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8s_primitives_35cls.pt")

# Mapping: primitive class name → equivalent standard duration type (if applicable)
# Note: The assembler will map "noteheadBlack" + "flag" -> "Eighth-note", etc.
# We keep these for the legacy pipeline compatibility after assembly.
NOTE_CLASSES = {
    'Full-note':      ('whole',    4),
    'Half-note':      ('half',     2),
    'Half-dot':       ('half',     3),  
    'Quarter-note':   ('quarter',  1),
    'Quarter-dot':    ('quarter',  1.5),  
    'Eighth-note':    ('eighth',   0.5),
    'Sixteenth-note': ('16th',     0.25),
}

# The primitive model rest names
REST_CLASSES = {
    'restWhole':    ('whole',    4),
    'restHalf':    ('half',     2),
    'restQuarter': ('quarter',  1),
    'restEighth':  ('eighth',   0.5),
    'rest16th':  ('16th',   0.25),
}

CLEF_CLASSES = {'clefG', 'clefF'}

ACCIDENTAL_CLASSES = {
    'accidentalFlat': 'flat', 
    'accidentalSharp': 'sharp', 
    'accidentalNatural': 'natural'
}

# Treble clef pitch mapping: staff_position → (step, octave)
TREBLE_PITCH_MAP = {
    -6: ('F', 3), -5: ('G', 3), -4: ('A', 3), -3: ('B', 3),
    -2: ('C', 4), -1: ('D', 4), 0:  ('E', 4), 1:  ('F', 4),
    2:  ('G', 4), 3:  ('A', 4), 4:  ('B', 4), 5:  ('C', 5),
    6:  ('D', 5), 7:  ('E', 5), 8:  ('F', 5), 9:  ('G', 5),
    10: ('A', 5), 11: ('B', 5), 12: ('C', 6), 13: ('D', 6), 14: ('E', 6),
}

# Bass clef pitch mapping
BASS_PITCH_MAP = {
    -6: ('A', 1), -5: ('B', 1), -4: ('C', 2), -3: ('D', 2),
    -2: ('E', 2), -1: ('F', 2), 0:  ('G', 2), 1:  ('A', 2),
    2:  ('B', 2), 3:  ('C', 3), 4:  ('D', 3), 5:  ('E', 3),
    6:  ('F', 3), 7:  ('G', 3), 8:  ('A', 3), 9:  ('B', 3),
    10: ('C', 4), 11: ('D', 4), 12: ('E', 4), 13: ('F', 4), 14: ('G', 4),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    class_id: int


@dataclass
class MusicalEvent:
    event_type: str      
    duration_type: str   
    duration_beats: float
    step: Optional[str] = None
    octave: Optional[int] = None
    alter: Optional[int] = None
    x_position: float = 0.0

@dataclass
class ChordEvent:
    notes: List[MusicalEvent]
    duration_beats: float
    duration_type: str
    is_chord: bool


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None

def load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        print(f"[PrimitiveYOLO] 📦 Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"[PrimitiveYOLO] ❌ Model file not found: {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
        print("[PrimitiveYOLO] ✅ Model loaded successfully")
    return _model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def run_detection(image_path: str, conf: float = 0.25, iou: float = 0.7) -> List[Detection]:
    model = load_model()
    print(f"[PrimitiveYOLO] 🔍 Running inference (conf={conf}, iou={iou}, imgsz=1280)...")
    results = model.predict(source=image_path, imgsz=1280, conf=conf, iou=iou, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        names = model.names
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = names[cls_id] if cls_id in names else f"class_{cls_id}"
            x_c, y_c, w, h = boxes.xywh[i].tolist()
            conf_val = boxes.conf[i].item()

            detections.append(Detection(
                class_name=cls_name,
                x_center=x_c, y_center=y_c,
                width=w, height=h,
                confidence=conf_val, class_id=cls_id,
            ))

    detections.sort(key=lambda d: d.x_center)
    print(f"[PrimitiveYOLO] 📋 Detected {len(detections)} primitives.")
    return detections

def run_detection_sahi(
    image_path: str, conf: float = 0.25, iou: float = 0.7,
    slice_size: int = 640, overlap_ratio: float = 0.25
) -> List[Detection]:
    try:
        from sahi.predict import get_sliced_prediction
        from sahi.models.yolov8 import Yolov8DetectionModel
    except ImportError:
        raise ImportError("sahi is not installed")
        
    with open("requirements.txt") as req:
        device = "cuda:0" if "onnxruntime-gpu" in req.read() else "cpu"
        
    detection_model = Yolov8DetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=conf,
        device=device,
    )

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio,
        postprocess_match_metric="IOU", postprocess_match_threshold=iou,
    )

    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        detections.append(Detection(
            class_name=pred.category.name,
            x_center=bbox.minx + bbox.width / 2,
            y_center=bbox.miny + bbox.height / 2,
            width=bbox.width, height=bbox.height,
            confidence=pred.score.value,
            class_id=pred.category.id,
        ))
    detections.sort(key=lambda d: d.x_center)
    return detections


def _get_pitch_map(clef_type: str) -> Dict[int, Tuple[str, int]]:
    if clef_type == 'clefF':
        return BASS_PITCH_MAP
    return TREBLE_PITCH_MAP

def map_note_pitch(note_y: float, staffs: List[StaffGroup], current_clef: str = 'clefG') -> Tuple[str, int]:
    staff = find_closest_staff(note_y, staffs)
    if staff is None: return ('C', 4)
    raw_position = y_to_staff_position(note_y, staff)
    position = round(raw_position)
    pitch_map = _get_pitch_map(current_clef)
    min_pos, max_pos = min(pitch_map.keys()), max(pitch_map.keys())
    position = max(min_pos, min(max_pos, position))
    return pitch_map[position]

def _find_accidental_for_note(note_det: Detection, all_detections: List[Detection], max_distance: float = 50.0) -> Optional[int]:
    for det in all_detections:
        if det.class_name in ACCIDENTAL_CLASSES:
            x_dist = note_det.x_center - det.x_center
            y_dist = abs(note_det.y_center - det.y_center)
            if 0 < x_dist < max_distance and y_dist < max(note_det.height, 40):
                if det.class_name == 'accidentalFlat': return -1
                if det.class_name == 'accidentalSharp': return 1
                if det.class_name == 'accidentalNatural': return 0
    return None

def extract_unpadded_measures_for_staff(staff_events: List[ChordEvent], barlines: List[int]) -> List[List[ChordEvent]]:
    bounds = barlines + [float('inf')]
    current_bound_idx = 0
    current_measure_events = []
    measures = []
    
    for ev in staff_events:
        x_pos = ev.notes[0].x_position
        while x_pos > bounds[current_bound_idx] and current_bound_idx < len(bounds) - 1:
            measures.append(current_measure_events)
            current_measure_events = []
            current_bound_idx += 1
        current_measure_events.append(ev)
        
    measures.append(current_measure_events)
    # Ensure there's a measure for each interval between barlines
    while len(measures) <= len(barlines):
        measures.append([])

    return measures

def apply_rhythm_enforcer(measures: List[List[ChordEvent]], target_beats: float) -> List[List[ChordEvent]]:
    synced_measures = []
    for m in measures:
        total_beats = sum(ev.duration_beats for ev in m)
        if total_beats < target_beats - 0.01:
            diff = target_beats - total_beats
            duration_type = 'quarter'
            if diff == 0.5: duration_type = 'eighth'
            elif diff == 0.25: duration_type = '16th'
            elif diff == 2.0: duration_type = 'half'
            elif diff >= 4.0: duration_type = 'whole'
            m.append(ChordEvent([MusicalEvent('rest', duration_type, diff, x_position=0.0)], diff, duration_type, False))
        elif total_beats > target_beats + 0.01:
            accum = 0.0
            clipped_m = []
            for ev in m:
                if accum + ev.duration_beats <= target_beats + 0.01:
                    clipped_m.append(ev)
                    accum += ev.duration_beats
                else:
                    remaining = target_beats - accum
                    if remaining > 0.01:
                        ev.duration_beats = remaining
                        clipped_m.append(ev)
                    break
            m = clipped_m
        synced_measures.append(m)
        
    return synced_measures

def detections_to_measures(detections: List[Detection], staffs: List[StaffGroup], staff_barlines: Dict[int, List[int]], gray_image: np.ndarray, dx_tolerance: float = 15.0, staves_per_system: int = 1, target_beats: float = 4.0) -> Dict[int, List[List[ChordEvent]]]:
    part_measures: Dict[int, List[List[ChordEvent]]] = {i: [] for i in range(staves_per_system)}
    staff_detections = {id(staff): [] for staff in staffs}
    orphaned_detections = []
    
    for det in detections:
        staff = find_closest_staff(det.y_center, staffs)
        if staff is not None:
            staff_detections[id(staff)].append(det)
        else:
            orphaned_detections.append(det)

    if orphaned_detections and not staffs:
        staff_detections['fallback'] = orphaned_detections
    
    for sys_idx in range(0, len(staffs), staves_per_system):
        system_staffs = staffs[sys_idx : min(sys_idx + staves_per_system, len(staffs))]
        sys_barlines = staff_barlines.get(id(system_staffs[0]), [])
        
        sys_unpadded_measures = {p_idx: [] for p_idx in range(staves_per_system)}
        
        for staff_idx, staff in enumerate(system_staffs):
            part_idx = staff_idx % staves_per_system
            group = staff_detections.get(id(staff), [])
            group.sort(key=lambda d: d.x_center)
            current_clef = 'clefG'
            
            staff_musical_events = []

            for det in group:
                if det.class_name in CLEF_CLASSES:
                    current_clef = det.class_name
                    continue

                if det.class_name in ACCIDENTAL_CLASSES:
                    continue

                if det.class_name in NOTE_CLASSES:
                    duration_type, duration_beats = NOTE_CLASSES[det.class_name]
                    note_y = det.y_center
                    step, octave = map_note_pitch(note_y, [staff], current_clef)
                    alter = _find_accidental_for_note(det, group)

                    staff_musical_events.append(MusicalEvent('note', duration_type, duration_beats, step, octave, alter, det.x_center))

                elif det.class_name in REST_CLASSES:
                    duration_type, duration_beats = REST_CLASSES[det.class_name]
                    staff_musical_events.append(MusicalEvent('rest', duration_type, duration_beats, x_position=det.x_center))

            staff_chord_events = []
            current_chord = []
            for ev in staff_musical_events:
                if ev.event_type == 'rest':
                    if current_chord:
                        staff_chord_events.append(ChordEvent(current_chord, current_chord[0].duration_beats, current_chord[0].duration_type, len(current_chord)>1))
                        current_chord = []
                    staff_chord_events.append(ChordEvent([ev], ev.duration_beats, ev.duration_type, False))
                else:
                    if not current_chord:
                        current_chord.append(ev)
                    else:
                        if abs(ev.x_position - current_chord[0].x_position) <= dx_tolerance:
                            current_chord.append(ev)
                        else:
                            staff_chord_events.append(ChordEvent(current_chord, current_chord[0].duration_beats, current_chord[0].duration_type, len(current_chord)>1))
                            current_chord = [ev]
            if current_chord:
                staff_chord_events.append(ChordEvent(current_chord, current_chord[0].duration_beats, current_chord[0].duration_type, len(current_chord)>1))

            sys_unpadded_measures[part_idx] = extract_unpadded_measures_for_staff(staff_chord_events, sys_barlines)

        # Remove completely empty ghost measures (intervals with NO notes in ANY staff of the system)
        num_intervals = len(sys_barlines) + 1
        valid_intervals = []
        for i in range(num_intervals):
            is_empty = True
            for p_idx in range(staves_per_system):
                if p_idx in sys_unpadded_measures and i < len(sys_unpadded_measures[p_idx]) and len(sys_unpadded_measures[p_idx][i]) > 0:
                    is_empty = False
                    break
            if not is_empty:
                valid_intervals.append(i)
                
        # Apply the rhythm padder to valid intervals, and append to full song part_measures
        for p_idx in range(staves_per_system):
            if p_idx not in sys_unpadded_measures: continue
            
            valid_m = []
            for i in valid_intervals:
                if i < len(sys_unpadded_measures[p_idx]):
                    valid_m.append(sys_unpadded_measures[p_idx][i])
                else:
                    valid_m.append([])
                    
            synced_m = apply_rhythm_enforcer(valid_m, target_beats)
            part_measures[p_idx].extend(synced_m)

    return part_measures

def events_to_musicxml(part_measures: Dict[int, List[List[ChordEvent]]], time_signature: str = "4/4", title: str = "Primitive YOLO Detection") -> str:
    divisions = 4
    def beats_to_duration(beats: float) -> int: return int(beats * divisions)

    score = Element('score-partwise', version='4.0')
    work = SubElement(score, 'work')
    SubElement(work, 'work-title').text = title
    part_list = SubElement(score, 'part-list')
    
    time_beats, time_type = time_signature.split('/')
    
    for p_idx in sorted(part_measures.keys()):
        score_part = SubElement(part_list, 'score-part', id=f'P{p_idx+1}')
        SubElement(score_part, 'part-name').text = f'Hand {p_idx+1}' if p_idx < 2 else f'Part {p_idx+1}'

    for p_idx in sorted(part_measures.keys()):
        part = SubElement(score, 'part', id=f'P{p_idx+1}')
        measures = part_measures[p_idx]
        
        for m_idx, measure_events in enumerate(measures):
            measure = SubElement(part, 'measure', number=str(m_idx + 1))
            if m_idx == 0:
                attributes = SubElement(measure, 'attributes')
                SubElement(attributes, 'divisions').text = str(divisions)
                key = SubElement(attributes, 'key')
                SubElement(key, 'fifths').text = '0'
                time_elem = SubElement(attributes, 'time')
                SubElement(time_elem, 'beats').text = time_beats
                SubElement(time_elem, 'beat-type').text = time_type
                clef = SubElement(attributes, 'clef')
                if p_idx == 1:
                    SubElement(clef, 'sign').text = 'F'
                    SubElement(clef, 'line').text = '4'
                else:
                    SubElement(clef, 'sign').text = 'G'
                    SubElement(clef, 'line').text = '2'

            for chord_event in measure_events:
                note_elem = SubElement(measure, 'note')
                if chord_event.notes[0].event_type == 'rest':
                    SubElement(note_elem, 'rest')
                    SubElement(note_elem, 'duration').text = str(beats_to_duration(chord_event.duration_beats))
                    SubElement(note_elem, 'type').text = chord_event.duration_type
                else:
                    for note_idx, event in enumerate(chord_event.notes):
                        if note_idx > 0:
                            note_elem = SubElement(measure, 'note')
                            SubElement(note_elem, 'chord')
                        pitch = SubElement(note_elem, 'pitch')
                        SubElement(pitch, 'step').text = event.step
                        if event.alter is not None:
                            SubElement(pitch, 'alter').text = str(event.alter)
                        SubElement(pitch, 'octave').text = str(event.octave)
                        SubElement(note_elem, 'duration').text = str(beats_to_duration(chord_event.duration_beats))
                        SubElement(note_elem, 'type').text = chord_event.duration_type
                        if chord_event.duration_type in ('half', 'quarter') and chord_event.duration_beats in (3, 1.5):
                            SubElement(note_elem, 'dot')
                        if event.alter is not None:
                            acc_elem = SubElement(note_elem, 'accidental')
                            if event.alter == -1:
                                acc_elem.text = 'flat'
                            elif event.alter == 1:
                                acc_elem.text = 'sharp'
                            else:
                                acc_elem.text = 'natural'

        if not measures:
            measure = SubElement(part, 'measure', number='1')
            attributes = SubElement(measure, 'attributes')
            SubElement(attributes, 'divisions').text = str(divisions)
            SubElement(SubElement(attributes, 'time'), 'beats').text = time_beats
            SubElement(SubElement(attributes, 'time'), 'beat-type').text = time_type
            clef = SubElement(attributes, 'clef')
            if p_idx == 1:
                SubElement(clef, 'sign').text = 'F'
                SubElement(clef, 'line').text = '4'
            else:
                SubElement(clef, 'sign').text = 'G'
                SubElement(clef, 'line').text = '2'
            note_elem = SubElement(measure, 'note')
            SubElement(note_elem, 'rest')
            SubElement(note_elem, 'duration').text = str(divisions * 4)
            SubElement(note_elem, 'type').text = 'whole'

    rough_string = tostring(score, encoding='unicode')
    try:
        parsed = minidom.parseString(rough_string)
        pretty_body = parsed.toprettyxml(indent="  ", encoding=None)
        lines = pretty_body.split('\n')
        if lines[0].startswith('<?xml'): lines = lines[1:]
        pretty_body = '\n'.join(lines)
    except Exception:
        pretty_body = rough_string
    return '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC\n  "-//Recordare//DTD MusicXML 4.0 Partwise//EN"\n  "http://www.musicxml.org/dtds/partwise.dtd">\n' + pretty_body


def save_diagnostic_image(image_path: str, detections: List[Detection], staffs: List[StaffGroup], staff_barlines: Dict[int, List[int]], output_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None: return output_path
    h, w = img.shape[:2]
    for staff in staffs:
        for y in staff.line_ys:
            cv2.line(img, (0, y), (w, y), (255, 100, 0), 1, cv2.LINE_AA)
        
        # Draw barlines
        bls = staff_barlines.get(id(staff), [])
        for bx in bls:
            cv2.line(img, (bx, staff.top-10), (bx, staff.bottom+10), (0, 0, 255), 2, cv2.LINE_AA)

    for det in detections:
        color = (0, 200, 0) if det.class_name in NOTE_CLASSES else (0, 140, 255) if det.class_name in REST_CLASSES else (200, 0, 200)
        x1, y1 = int(det.x_center - det.width / 2), int(det.y_center - det.height / 2)
        x2, y2 = int(det.x_center + det.width / 2), int(det.y_center + det.height / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(output_path, img)
    return output_path

# ---------------------------------------------------------------------------
# MAIN API
# ---------------------------------------------------------------------------
def generation_workflow_primitive_yolo(
    image_path: str, output_dir: str = "./", conf: float = 0.25, iou: float = 0.7,
    dx_tolerance: float = 15.0, enable_beam_correction: bool = True,
    use_sahi: bool = False, sahi_slice_size: int = 640, sahi_overlap: float = 0.25,
    staves_per_system: int = 1, time_signature: str = "4/4"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    staffs = detect_staff_lines(image_path)
    
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if use_sahi:
        raw_detections = run_detection_sahi(image_path, conf=conf, iou=iou, slice_size=sahi_slice_size, overlap_ratio=sahi_overlap)
    else:
        raw_detections = run_detection(image_path, conf=conf, iou=iou)
        
    interline = 10.0
    if len(staffs) > 0 and len(staffs[0].line_ys) >= 2:
        interline = staffs[0].line_ys[1] - staffs[0].line_ys[0]

    staff_barlines = detect_barlines(image_path, staffs, raw_detections)

    # ASSEMBLE PRIMITIVES GEOMETRICALLY
    print("[PrimitiveYOLO] 🧩 Assembling primitives...")
    assembled_detections = assemble_primitives(
        raw_detections, 
        gray_image, 
        interline, 
        enable_beam_correction=enable_beam_correction
    )
    
    # target beats matching time signature e.g., '3/4' -> 3.0 beats
    tb_num, tb_den = map(int, time_signature.split('/'))
    target_beats = float(tb_num) * (4.0 / tb_den)
    
    part_measures = detections_to_measures(
        assembled_detections, staffs, staff_barlines, gray_image, 
        dx_tolerance=dx_tolerance, staves_per_system=staves_per_system, target_beats=target_beats
    )
    xml_str = events_to_musicxml(part_measures, time_signature=time_signature)
    
    xml_path = os.path.join(output_dir, "output.musicxml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
        
    save_diagnostic_image(image_path, assembled_detections, staffs, staff_barlines, os.path.join(output_dir, "bboxes.png"))
    return os.path.abspath(xml_path)
