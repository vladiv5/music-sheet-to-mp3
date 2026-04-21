# Music Sheet to MP3 (OMR & Audio Synthesis)

## Project Goal
The ultimate goal of this project is to build a complete, accessible, and cross-platform application that can read digital images of sheet music (Optical Music Recognition - OMR) and synthesize them into playable audio files (MP3/WAV). It aims to bridge the gap between visual musical notation and auditory feedback.

## Current State & Foundation
This repository currently contains the MVP (Minimum Viable Product) of the application. To establish a rapid testing foundation and validate the technical feasibility, the initial boilerplate and integration code were generated with the assistance of an AI. 

The system is built with a modular architecture:
* **Vision Module:** Uses the `oemer` library to extract musical symbols and structural data from raw images, converting them into standard `MusicXML` format.
* **Audio Synthesis Module:** Uses `music21` and `midi2audio` (powered by FluidSynth) to parse the XML, translate it to MIDI, and render it into an audio file using a SoundFont.
* **Dockerized Environment:** The entire pipeline runs exclusively inside a Docker container. This guarantees 100% portability, eliminates local dependency conflicts, and allows for seamless CPU/GPU testing without polluting the host machine.

## Testing
A preliminary test was conducted using a clean digital sheet of "Für Elise" (available in the `tests/test1` folder). The pipeline successfully executed from end-to-end, taking the `.png` image and producing a playable `.wav` file. 
Going forward, all active test outputs are isolated in the `tests/current_test` directory to maintain a clean root workspace.

## Known Limitations & Future Work
While the MVP successfully maps notes to their correct pitch, there are significant limitations in the current OMR model:
* **Rhythm and Tempo Issues:** The AI struggles to accurately interpret note durations (e.g., confusing quarter notes with sixteenth notes), leading to erratic pacing and unnatural pauses during playback.
* **Complex Notation:** Handwritten annotations, complex ties, or non-standard measure lines can cause the extraction pipeline to fail.

**Next Steps:**
1. Custom Model Training: Replace the underlying OMR engine by training or fine-tuning custom models (e.g., YOLO architectures on datasets like DeepScores) to compare accuracy and improve rhythm detection.
2. Graphical User Interface (GUI): Implement a web-based interface using `Streamlit` to move away from the CLI and allow users to easily upload images and play audio directly in the browser.

**Next Steps / Roadmap:**
1. **Custom Model Training:** Replace the underlying OMR engine by training or fine-tuning custom state-of-the-art models (e.g., YOLO architectures on datasets like DeepScores) to drastically improve rhythm and beam detection.
2. **Baseline Engine Comparison:** Preserve the current `oemer` integration in the project structure as a permanent baseline. This will allow for direct, side-by-side A/B testing between the pre-packaged library and the newly trained custom models.
3. **PDF Processing Optimization:** Resolve the current Docker memory/volume crash (`unexpected EOF`) when parsing and converting high-resolution multi-page PDFs using `poppler`.
4. **Storage Management:** Safely map the Docker internal `/tmp` directory to a secondary host drive (e.g., `D:`) to prevent the primary OS drive from bloating during heavy AI inference tasks.

## Major Refactoring & Upgrade (2026-04-01)

The project has undergone a significant architectural upgrade to support faster development and experimental AI models:

*   **Hot-Reloading in Docker:** Enabled real-time UI and logic updates. By mounting `app.py` and the `core/` folder as volumes, changes saved locally are instantly reflected in the running Streamlit container without requiring a Docker rebuild.
*   **AI Engine Dispatcher:** Implemented a selection system in the UI to toggle between the **Oemer Baseline** and the new **Custom CV Model** (YOLO-based skeleton).
*   **Benchmarking System:** Added a high-precision timer to measure the exact inference time of each AI engine. This "Processing Time" is displayed as a metric in the UI for A/B testing and performance optimization.
*   **Custom CV Model Skeleton:** Created `models/custom_yolo_inference.py` as a functional mock. It simulates an AI inference pipeline (Do-Re-Mi scale) to validate the integration before plugging in a real YOLO-based model.
*   **Enhanced UI (Streamlit):**
    *   Added **MusicXML Parsing** via `music21` to display the extracted note sequence in a collapsible expander.
    *   Improved visual feedback with specialized metrics and success/info notifications.
*   **Clean Repository Management:** Updated `.gitignore` to prevent large binary artifacts (WAV/MID) and temporary Docker data from cluttering the repository, while preserving historical tests and inputs.

## First Real YOLO Training Experiment (2026-04-10)

With the A/B testing pipeline in place, the next logical step was to replace the mock Custom CV Engine with a real trained model — even a basic one — to validate the full end-to-end workflow from data collection to inference.

### Dataset Collection & Labeling

*   **5 music sheet images** were manually selected and downloaded (simple, single-voice pieces suitable for a Nivel 0 curriculum).
*   The images were uploaded to **Roboflow** and annotated by hand with 7 musical symbol classes:
    `barline`, `clef_f`, `clef_g`, `note_half`, `note_quarter`, `note_whole`, `time_signature`
*   The labeled dataset was exported in YOLO format and saved locally under `dataset/minor_dataset/dataset_1/`.

### Training

*   A **YOLOv8n** (nano) model was fine-tuned on this 5-image dataset using `core/train_yolo.py`, running inside the Docker container.
*   Training was performed for **50 epochs** at `imgsz=640`.
*   The `docker-compose.yml` was updated with `shm_size: '4gb'` to prevent shared memory crashes during training on GPU.
*   The trained weights were saved to `tests/current_test/runs/omr_nivel06/weights/best.pt`.

### Inference & Results

*   A dedicated inference script `core/test_ochi.py` was written to run the trained model against the training images with a **very low confidence threshold (5%)** to force the model to surface even its weakest detections.
*   Two inference runs were executed:
    *   **`rezultate_test`** — first run (default confidence), results copied from the container via `docker cp`.
    *   **`rezultate_test2`** — second run at `conf=0.05`, exposing more detections for visual inspection.

### Observations & Conclusion

The model's detection quality is **intentionally poor** — this is expected and acceptable at this stage. The goal of this experiment was **not** to build an accurate model, but to:

1.  ✅ Validate that the complete pipeline works (label → train → infer → inspect results).
2.  ✅ Confirm that the trained weights can be loaded and run inside the Docker environment.
3.  ✅ Establish a repeatable workflow for future iterations with larger, richer datasets.

The next step is to scale the dataset (more images, more diverse annotations) and retrain to meaningfully improve detection accuracy.

## Pitch Detection & Staff Isolation (2026-04-14)

Following the initial YOLO training, the project focus shifted to accurately extracting musical meaning from raw bounding boxes.

*   **Classical CV Staff Detection:** Implemented `core/staff_detector.py`, a pure OpenCV module that detects the horizontal staff grid using morphological operations and row-projection profiles. It uniquely identifies systems of 5 lines and computes spatial measurements (e.g., inter-line pacing).
*   **Pitch Mapping Strategy:** By isolating the staff framework, YOLO bounding-box Y-coordinates can be mapped natively into precise musical pitches, taking interline spacing and ledger lines into account.
*   **Blob Extraction Pipeline:** Actively implementing shape-based notehead detection to effectively separate isolated noteheads from stems or overlapping staff lines.

## Deep Primitive Inference & Polyphonic Synchronization (2026-04-21)

Today's session marked a massive leap forward into full polyphony support and mathematical rhythm synchronization. I integrated the new custom-trained `YOLOv8s` model into the main pipeline and solved several critical OMR architecture challenges.

### The 35-Class Primitive Model Rationale
Originally, datasets like DeepScores contain over 100 distinct classes, treating every variation of a musical symbol (e.g., "quarter note stem up", "eighth note stem down") as a unique class. I decided to discard this approach. Instead, I carefully distilled the dataset down to just 35 "primitive" building blocks (e.g., `noteheadBlack`, `flag8thUp`, `beam`, `augmentationDot`). 
The reason for this choice is cognitive load: it is significantly easier and more accurate for the neural network to detect fundamental shapes rather than memorizing every possible composite permutation. I then shifted the burden of understanding composites to algorithmic logic.

### 1. Primitive Assembly Engine
Instead of hoping the neural network perfectly reads complete objects right away, I built `core/primitive_assembler.py` which takes the raw primitives output by my YOLO model and mathematically groups them together based on proximity. This constructs cohesive musical objects ready for XML mapping deterministically.
*   **Beam Parsing:** If a notehead's X-axis intersects a detected horizontal `beam` bounding box, its duration is dynamically shifted to `eighth` (1 beam overlap) or `16th` (2+ beam overlaps).

### 2. Multi-Stave/Polyphony (Grand Staff Support)
I overhauled the XML assembler in `models/primitive_yolo_inference.py` to support parallel hand playback. 
*   Added a `Staves per System` UI slider. 
*   By setting it to `2` (Piano), the model segregates notes on the top staves (odd index) as Part 1 (Right Hand, forced to Treble Clef) and bottom staves (even index) as Part 2 (Left Hand, forced to Bass Clef). They are emitted into sibling `<part>` streams in MusicXML, ensuring both hands synthesize perfectly simultaneously.

### 3. Spatial Barline Detection (Computer Vision)
A major vulnerability in sequential pipeline playback was rhythmic drift. If YOLO missed a left-hand note, the left-hand track would shift out of sync with the right hand. To fix this, I created `core/barline_detector.py`:
*   Applied a vertical Otsu binarization and `cv2.morphologyEx(MORPH_OPEN)` with a vertical kernel height of `3.5 * interline` to extract bar lines visually without YOLO.
*   **Stem Filtering Algorithm:** To prevent false positives where note stems are mistaken as barlines, I applied an intersection filter matrix: if a vertical line is shorter than `6.0 * interline` and its X-center is within `< 0.85 * interline` of a raw `notehead`, it is discarded as a stem. Tall barlines (e.g., `h > 10 * interline`) bridging multiple staves bypass this filter.

### 4. The Rhythm Enforcer & Ghost Measure Pruner
I modified `_split_into_measures` to slice notes spatially using the X-coordinates provided by the barline detector. I feed the Time Signature (e.g., `3/4`) into the UI:
*   Formula: `target_beats = (numerator) * (4.0 / denominator)`
*   When a spatial interval (a measure bounded by two vertical lines) lacks the required beats due to YOLO missing a note, the Enforcer mathematically injects a padding `<rest>` element at the end of the measure, bringing the beat count to exact synchronization.
*   **Ghost Measures:** Empty spatial arrays left before the first barline or after the last barline are algorithmically pruned if no hand has any notes within that interval, avoiding hallucinated silence blocks.

### Current Limitations & Next Roadmap
Testing this mathematical rigidity on highly complex, dense arrangements exposed classic limitations:
1.  **NMS Crowding:** Dense 16th-note chords cause YOLO to skip detecting small noteheads due to bounding-box overlaps causing Non-Maximum Suppression (NMS) collisions. Without notehead anchors, the stems fail the OpenCV stem filter and are falsely declared tracking barlines.
2.  **Proposed Fix:** During the next session, I will strictly integrate **SAHI (Slicing Aided Hyper Inference)**. By dividing the image into overlapping ~640x640 patches, I will forcefully upscale dense note clusters, significantly increasing recall without overwhelming the YOLO architecture.
