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