"""
core/custom_yolo_inference.py
------------------------------
SKELETON / MOCK — Custom Computer Vision Inference Engine

This module acts as the integration point for a future custom-trained model.
Currently it simulates the inference pipeline without any real AI processing.

Architecture note:
  The function signature is intentionally identical to generation_workflow_oemer()
  so that app.py can swap engines transparently via a single dispatcher block.
  When the real model is ready, replace the body of this function with actual
  inference logic — the rest of the codebase stays untouched.
"""

import os
import time


# --- Hardcoded MusicXML template: C Major scale (Do-Re-Mi-Fa-Sol-La-Si)  ---
# MusicXML 4.0 — minimal valid document, compatible with music21 & MuseScore.
_MUSICXML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC
  "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work>
    <work-title>Custom CV Model — Demo Output (C Major Scale)</work-title>
  </work>
  <identification>
    <encoding>
      <software>Music Sheet to MP3 — Custom CV Engine (Mock)</software>
    </encoding>
  </identification>
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <!-- Do (C4) -->
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <!-- Re (D4) -->
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <!-- Mi (E4) -->
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <!-- Fa (F4) -->
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <!-- Sol (G4) -->
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <!-- La (A4) -->
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <!-- Si (B4) -->
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <!-- Do octavă superioară (C5) -->
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>
"""


def generation_workflow_custom_yolo(image_path: str, output_dir: str = "./") -> str:
    """
    MOCK inference function — mimics the Oemer baseline interface.

    Steps simulated:
      1. Log the received image path (proves the dispatcher reached this engine).
      2. Sleep 2 seconds to simulate actual model inference latency.
      3. Write a hardcoded MusicXML (C Major scale) to output_dir.
      4. Return the absolute path to the generated .musicxml file.

    When the real custom model is ready:
      Replace steps 2-3 with actual CV/ML inference that reads `image_path`
      and writes a genuine MusicXML to `output_dir`. The function signature
      and return contract must stay the same.

    Args:
        image_path: Absolute or relative path to the input sheet music image.
        output_dir: Directory where the output .musicxml file will be saved.

    Returns:
        Absolute path to the generated .musicxml file.

    Raises:
        RuntimeError: If the output directory cannot be written.
    """
    print(f"[CustomCVEngine] 🟡 Inference started — input image: {image_path}")
    print(f"[CustomCVEngine] 📂 Output directory: {output_dir}")

    # --- Step 1: Ensure the output directory exists ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 2: Simulate inference time (replace with real model call) ---
    print("[CustomCVEngine] ⏳ Simulating model inference (2s) ...")
    time.sleep(2)

    # --- Step 3: Write the hardcoded MusicXML to disk ---
    output_xml_path = os.path.join(output_dir, "custom_cv_output.musicxml")
    try:
        with open(output_xml_path, "w", encoding="utf-8") as f:
            f.write(_MUSICXML_TEMPLATE)
    except OSError as e:
        raise RuntimeError(
            f"[CustomCVEngine] ❌ Could not write MusicXML to {output_xml_path}: {e}"
        ) from e

    print(f"[CustomCVEngine] ✅ Mock MusicXML written to: {output_xml_path}")
    return output_xml_path
