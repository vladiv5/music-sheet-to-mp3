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
