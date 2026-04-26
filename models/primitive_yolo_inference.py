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
from core.octave_detector import detect_octave_shifts
from core.barline_detector import detect_barlines

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use the new primitive model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8s_primitives_35cls.pt")
BARLINE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "training", "runs", "detect", "runs", "barline_yolov8n", "weights", "best.pt")

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
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[PrimitiveYOLO] 📦 Loading model from: {MODEL_PATH} on {device}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"[PrimitiveYOLO] ❌ Model file not found: {MODEL_PATH}")
        _model = YOLO(MODEL_PATH).to(device)
        print(f"[PrimitiveYOLO] ✅ Model loaded successfully on {device}")
    return _model

_barline_model = None
def load_barline_model():
    global _barline_model
    if _barline_model is None:
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[PrimitiveYOLO] 📦 Loading Barline model from: {BARLINE_MODEL_PATH} on {device}")
        if not os.path.exists(BARLINE_MODEL_PATH):
            raise FileNotFoundError(f"[PrimitiveYOLO] ❌ Barline model file not found: {BARLINE_MODEL_PATH}")
        _barline_model = YOLO(BARLINE_MODEL_PATH).to(device)
        print(f"[PrimitiveYOLO] ✅ Barline Model loaded successfully on {device}")
    return _barline_model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def run_detection(image_path: str, conf: float = 0.25, iou: float = 0.7) -> List[Detection]:
    model = load_model()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PrimitiveYOLO] 🔍 Running inference (conf={conf}, iou={iou}, imgsz=1280, device={device})...")
    results = model.predict(source=image_path, imgsz=1280, conf=conf, iou=iou, device=device, verbose=False)

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

def run_barline_detection(image_path: str, conf: float = 0.5, iou: float = 0.5) -> List[Detection]:
    model = load_barline_model()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model.predict(source=image_path, imgsz=1280, conf=conf, iou=iou, device=device, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            x_c, y_c, w, h = boxes.xywh[i].tolist()
            conf_val = boxes.conf[i].item()
            detections.append(Detection(
                class_name="barline", x_center=x_c, y_center=y_c,
                width=w, height=h, confidence=conf_val, class_id=0
            ))
            
    return sorted(detections, key=lambda d: d.x_center)

def detect_barlines_ai(image_path: str, staffs: List[StaffGroup], conf: float = 0.25) -> Dict[int, List[int]]:
    barline_dets = run_barline_detection(image_path, conf=conf)
    staff_barlines = {}
    
    for staff in staffs:
        staff_bls = []
        for det in barline_dets:
            # Check if barline intersects staff vertically
            bl_top = det.y_center - det.height / 2
            bl_bottom = det.y_center + det.height / 2
            
            if bl_top < staff.bottom and bl_bottom > staff.top:
                staff_bls.append(int(det.x_center))
        staff_bls = sorted(list(set(staff_bls)))
        
        filtered = []
        for bl in staff_bls:
            if not filtered or bl - filtered[-1] > 20:
                filtered.append(bl)
        staff_barlines[id(staff)] = filtered
        
    return staff_barlines

def run_detection_sahi(
    image_path: str, conf: float = 0.25, iou: float = 0.7,
    slice_size: int = 640, overlap_ratio: float = 0.25
) -> List[Detection]:
    try:
        from sahi.predict import get_sliced_prediction
        from sahi.models.ultralytics import UltralyticsDetectionModel
    except ImportError:
        raise ImportError("sahi is not installed")
        
    with open("requirements.txt") as req:
        device = "cuda:0" if "onnxruntime-gpu" in req.read() else "cpu"
        
    detection_model = UltralyticsDetectionModel(
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
        b_w = bbox.maxx - bbox.minx
        b_h = bbox.maxy - bbox.miny
        detections.append(Detection(
            class_name=pred.category.name,
            x_center=bbox.minx + b_w / 2,
            y_center=bbox.miny + b_h / 2,
            width=b_w, height=b_h,
            confidence=pred.score.value,
            class_id=pred.category.id,
        ))
    detections.sort(key=lambda d: d.x_center)
    return detections

def bb_iou(d1: Detection, d2: Detection) -> float:
    x1_min, y1_min = d1.x_center - d1.width/2, d1.y_center - d1.height/2
    x1_max, y1_max = d1.x_center + d1.width/2, d1.y_center + d1.height/2
    x2_min, y2_min = d2.x_center - d2.width/2, d2.y_center - d2.height/2
    x2_max, y2_max = d2.x_center + d2.width/2, d2.y_center + d2.height/2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    if inter_area == 0: return 0.0
    
    d1_area = d1.width * d1.height
    d2_area = d2.width * d2.height
    return inter_area / float(d1_area + d2_area - inter_area)

def apply_musical_nms(detections: List[Detection], interline: float, iou_thresh: float = 0.5) -> List[Detection]:
    # I sort the detections by confidence in descending order to keep the strongest ones
    detections.sort(key=lambda d: d.confidence, reverse=True)
    kept = []
    
    for d in detections:
        keep = True
        for k in kept:
            # I only apply suppression if the primitive classes match exactly
            if d.class_name == k.class_name:
                overlap = bb_iou(d, k)
                
                if overlap > iou_thresh:
                    # I created a special rule for chords (noteheads):
                    # If I detect overlapping noteheads, I check their vertical distance
                    if 'note' in d.class_name.lower() or 'head' in d.class_name.lower():
                        y_dist = abs(d.y_center - k.y_center)
                        
                        # If the vertical distance is at least ~0.4 of an interline, 
                        # I assume it is a distinct note in a chord (like a second interval) 
                        # rather than a duplicate bounding box, so I don't suppress it!
                        if y_dist >= interline * 0.4:
                            continue 
                    
                    # I suppress the detection if it didn't pass my chord exception
                    keep = False
                    break
                    
        if keep: 
            # I permanently keep the valid detection
            kept.append(d)
            
    return kept

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

def get_key_alterations(fifths: int) -> Dict[str, int]:
    sharps_order = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    flats_order = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
    alterations = {}
    if fifths > 0:
        for i in range(min(fifths, 7)):
            alterations[sharps_order[i]] = 1
    elif fifths < 0:
        for i in range(min(-fifths, 7)):
            alterations[flats_order[i]] = -1
    return alterations

def infer_key_signature(staff_detections: List[Detection]) -> int:
    first_note_x = float('inf')
    for det in staff_detections:
        if det.class_name in NOTE_CLASSES or det.class_name in REST_CLASSES:
            if det.x_center < first_note_x:
                first_note_x = det.x_center
    sharps = 0
    flats = 0
    for det in staff_detections:
        if det.x_center < first_note_x:
            if det.class_name == 'accidentalSharp':
                sharps += 1
            elif det.class_name == 'accidentalFlat':
                flats += 1
    if sharps > 0 and flats == 0:
        return min(sharps, 7)
    elif flats > 0 and sharps == 0:
        return -min(flats, 7)
    return 0

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

def detections_to_measures(detections: List[Detection], staffs: List[StaffGroup], staff_barlines: Dict[int, List[int]], gray_image: np.ndarray, dx_tolerance: float = 15.0, staves_per_system: int = 1, target_beats: float = 4.0, inherited_fifths: int = None, octave_shift: int = 0, octave_shifts: list = None) -> Tuple[Dict[int, List[List[ChordEvent]]], int]:
    
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
        
    # Infer global key signature (Majority voting)
    all_fifths = []
    for staff_id, group in staff_detections.items():
        if staff_id != 'fallback':
            f = infer_key_signature(group)
            if f != 0:
                all_fifths.append(f)
    
    # --- LOGICA NOUĂ CORECTATĂ ---
    if inherited_fifths is not None:
        global_fifths = inherited_fifths
        print(f"[PrimitiveYOLO] 🔗 Using INHERITED Key Signature: {global_fifths} fifths")
    else:
        global_fifths = 0
        if all_fifths:
            global_fifths = max(set(all_fifths), key=all_fifths.count)
        print(f"[PrimitiveYOLO] 🔑 Inferred NEW Key Signature: {global_fifths} fifths")
    # -----------------------------
        
    key_alts = get_key_alterations(global_fifths)
    
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
                    octave += octave_shift
                    
                    if octave_shifts:
                        for shift in octave_shifts:
                            if shift['staff_id'] == id(staff) and shift['x_start'] <= det.x_center <= shift['x_end']:
                                octave += shift['amount']
                                break
                    alter = _find_accidental_for_note(det, group)
                    if alter is None:
                        alter = key_alts.get(step, None)

                    staff_musical_events.append(MusicalEvent('note', duration_type, duration_beats, step, octave, alter, det.x_center))

                elif det.class_name in REST_CLASSES:
                    duration_type, duration_beats = REST_CLASSES[det.class_name]
                    staff_musical_events.append(MusicalEvent('rest', duration_type, duration_beats, x_position=det.x_center))

            staff_chord_events = []
            current_chord = []
            for ev in staff_musical_events:
                if not current_chord:
                    current_chord.append(ev)
                else:
                    if abs(ev.x_position - current_chord[0].x_position) <= dx_tolerance:
                        current_chord.append(ev)
                    else:
                        notes_only = [e for e in current_chord if e.event_type == 'note']
                        if notes_only:
                            staff_chord_events.append(ChordEvent(notes_only, notes_only[0].duration_beats, notes_only[0].duration_type, len(notes_only)>1))
                        else:
                            staff_chord_events.append(ChordEvent([current_chord[0]], current_chord[0].duration_beats, current_chord[0].duration_type, False))
                        current_chord = [ev]
                        
            if current_chord:
                notes_only = [e for e in current_chord if e.event_type == 'note']
                if notes_only:
                    staff_chord_events.append(ChordEvent(notes_only, notes_only[0].duration_beats, notes_only[0].duration_type, len(notes_only)>1))
                else:
                    staff_chord_events.append(ChordEvent([current_chord[0]], current_chord[0].duration_beats, current_chord[0].duration_type, False))

            sys_unpadded_measures[part_idx] = extract_unpadded_measures_for_staff(staff_chord_events, sys_barlines)

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

    return part_measures, global_fifths

def events_to_musicxml(part_measures: Dict[int, List[List[ChordEvent]]], time_signature: str = "4/4", title: str = "Primitive YOLO Detection", fifths: int = 0) -> str:
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
                SubElement(key, 'fifths').text = str(fifths)
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


def save_diagnostic_image(image_path: str, detections: List[Detection], staffs: List[StaffGroup], staff_barlines: Dict[int, List[int]], output_path: str, measure_crops: list = None) -> str:
    img = cv2.imread(image_path)
    if img is None: return output_path
    h, w = img.shape[:2]

    # Draw measure crop rectangles (Measure-SAHI visualization) — semi-transparent teal overlay
    if measure_crops:
        overlay = img.copy()
        for (cx1, cy1, cx2, cy2) in measure_crops:
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 210, 210), -1)
        cv2.addWeighted(overlay, 0.13, img, 0.87, 0, img)
        for idx, (cx1, cy1, cx2, cy2) in enumerate(measure_crops):
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 180, 180), 2)
            cv2.putText(img, f"M{idx+1}", (cx1 + 4, cy1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 120), 1)

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
    staves_per_system: int = 1, time_signature: str = "4/4",
    use_ai_barlines: bool = False, use_system_sahi: bool = False,
    inherited_fifths: int = None, sahi_systems_per_slice: int = 1,
    octave_shift: int = 0
):
    os.makedirs(output_dir, exist_ok=True)
    staffs = detect_staff_lines(image_path)
    
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(image_path)
    
    interline = 10.0
    interline = 10.0
    if len(staffs) > 0 and len(staffs[0].line_ys) >= 2:
        interline = staffs[0].line_ys[1] - staffs[0].line_ys[0]

    # Detect local octave shifts (8va, 8vb) using OCR
    print("[PrimitiveYOLO] 🔍 Scanning for local octave shifts (8va/8vb)...")
    t0_ocr = time.time()
    octave_shifts = detect_octave_shifts(image_path, staffs, interline)
    print(f"[PrimitiveYOLO] ⏱️ OCR Octave Detection took {time.time() - t0_ocr:.2f}s")
    if octave_shifts:
        print(f"[PrimitiveYOLO] 🎯 Found {len(octave_shifts)} octave shift markers.")

    if use_system_sahi:
        # ── SYSTEM-SAHI (Grupare pe portative) ──────────────────────────────────
        print("[PrimitiveYOLO] ✂️  System-SAHI: Slicing by groups of staffs...")
        
        # Mai întâi detectăm barlines-urile pentru că avem nevoie de ele mai târziu în pipeline, 
        # chiar dacă acum nu mai tăiem imaginea după ele!
        print("[PrimitiveYOLO] 🎼 Detecting barlines via AI Nano model...")
        try:
            staff_barlines = detect_barlines_ai(image_path, staffs, conf=0.15)
        except Exception as e:
            print(f"[PrimitiveYOLO] Warning: AI barline detection failed ({e}). Falling back to OpenCV.")
            staff_barlines = detect_barlines(image_path, staffs, [])

        raw_detections = []
        system_crop_rects = []   # Aici salvăm coordonatele pentru vizualizarea finală
        img_h, img_w = color_image.shape[:2]
        model = load_model()

        CANVAS_SIZE = 1280
        # Calculăm câte portative (staves) intră într-o felie, bazat pe numărul de sisteme cerut
        chunk_size = staves_per_system * sahi_systems_per_slice 

        # Iterăm prin portative, grupându-le în funcție de setările utilizatorului
        for i in range(0, len(staffs), chunk_size):
            group = staffs[i : min(i + chunk_size, len(staffs))]
            
            # Tăiem exact la jumătatea distanței dintre sisteme pentru a EVITA SUPRAPUNEREA total!
            # Astfel, nicio notă nu este procesată de două ori și nu forțăm NMS-ul să facă minuni.
            if i == 0:
                y1 = 0
            else:
                prev_staff = staffs[i - 1]
                curr_staff = group[0]
                y1 = int((prev_staff.bottom + curr_staff.top) / 2)
                
            if i + chunk_size >= len(staffs):
                y2 = img_h
            else:
                curr_bottom_staff = group[-1]
                next_top_staff = staffs[i + chunk_size]
                y2 = int((curr_bottom_staff.bottom + next_top_staff.top) / 2)
            
            if y2 - y1 < 10:
                continue

            # Salvăm coordonatele pentru a le desena pe poza de diagnostic la final
            system_crop_rects.append((0, y1, img_w, y2))
            
            # Decupăm felia orizontală (luăm TOATĂ lățimea paginii)
            slice_crop = color_image[y1:y2, 0:img_w]
            
            # Trimitem felia la YOLO. Va face resize păstrând aspect ratio-ul.
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = model.predict(
                source=slice_crop, imgsz=CANVAS_SIZE,
                conf=conf, iou=iou, device=device, verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                names = model.names
                for k in range(len(boxes)):
                    cls_id = int(boxes.cls[k].item())
                    cls_name = names[cls_id] if cls_id in names else f"class_{cls_id}"
                    x_c_local, y_c_local, w_det, h_det = boxes.xywh[k].tolist()
                    conf_val = boxes.conf[k].item()

                    # Transformăm coordonatele Y înapoi la scara paginii globale (X rămâne la fel)
                    raw_detections.append(Detection(
                        class_name=cls_name,
                        x_center=x_c_local,
                        y_center=y_c_local + y1, # Adăugăm offset-ul feliei
                        width=w_det, height=h_det,
                        confidence=conf_val, class_id=cls_id,
                    ))

        # Folosim NMS-ul tău muzical pe toate detecțiile la un loc pentru a șterge dublurile
        # care s-ar fi putut forma la linia de tăietură dintre 2 felii.
        raw_detections = apply_musical_nms(raw_detections, interline, iou_thresh=iou)
        print(f"[PrimitiveYOLO] 📋 System-SAHI: {len(raw_detections)} detections after NMS.")
        
        # Facem un truc: redenumim lista noastră ca funcția de desenare să creadă că sunt "măsuri"
        # și să le deseneze transparent pe imaginea finală din folder.
        measure_crop_rects = system_crop_rects

    elif not use_ai_barlines:
        # ── STANDARD WHOLE-PAGE + OPENCV BARLINES ─────────────────────────────────
        print("[PrimitiveYOLO] 📐 Using original whole-page inference & OpenCV barlines...")
        if use_sahi:
            raw_detections = run_detection_sahi(image_path, conf=conf, iou=iou, slice_size=sahi_slice_size, overlap_ratio=sahi_overlap)
        else:
            raw_detections = run_detection(image_path, conf=conf, iou=iou)
        staff_barlines = detect_barlines(image_path, staffs, raw_detections)
    else:
        # ── WHOLE-PAGE + AI BARLINES ───────────────────────────────────────────────
        print("[PrimitiveYOLO] 📏 Using original whole-page inference & AI Nano Barlines...")
        if use_sahi:
            raw_detections = run_detection_sahi(image_path, conf=conf, iou=iou, slice_size=sahi_slice_size, overlap_ratio=sahi_overlap)
        else:
            raw_detections = run_detection(image_path, conf=conf, iou=iou)
            
        try:
            staff_barlines = detect_barlines_ai(image_path, staffs, conf=0.15)
        except Exception as e:
            print(f"[PrimitiveYOLO] Warning: AI Barline detection failed ({e}). Falling back to algorithmic detection.")
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
    
    # --- AICI TRANSMITEM parametrul mai departe ---
    part_measures, fifths = detections_to_measures(
        assembled_detections, staffs, staff_barlines, gray_image, 
        dx_tolerance=dx_tolerance, staves_per_system=staves_per_system, target_beats=target_beats,
        inherited_fifths=inherited_fifths, octave_shift=octave_shift,
        octave_shifts=octave_shifts
    )
    xml_str = events_to_musicxml(part_measures, time_signature=time_signature, fifths=fifths)
    
    xml_path = os.path.join(output_dir, f"output_{os.path.basename(image_path)}.musicxml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
        
    diag_name = f"bboxes_{os.path.basename(image_path)}"
    save_diagnostic_image(
        image_path, assembled_detections, staffs, staff_barlines,
        os.path.join(output_dir, diag_name),
        measure_crops=locals().get('measure_crop_rects', None)
    )
    
    # --- AICI RETURNĂM TUPLUL ---
    return os.path.abspath(xml_path), fifths
