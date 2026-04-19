import os
import shutil
import time
import streamlit as st
from music21 import converter, note, chord
from pdf2image import convert_from_path
from core.image_processing import generation_workflow_oemer
from models.custom_yolo_inference import generation_workflow_custom_yolo
from core.audio_synthesis import convert_xml_to_mp3

# I configure the basic settings for my web page
st.set_page_config(page_title="Music Sheet to MP3", page_icon="🎵", layout="wide")

st.title("🎵 OMR Audio Synthesizer")
st.markdown("Upload a clean image or PDF of a music sheet and let the AI convert it into a playable audio file.")

# I updated the uploader to accept PDF files as well
uploaded_file = st.file_uploader("Upload your sheet music (PDF/PNG/JPG)", type=["png", "jpg", "jpeg", "pdf"])

soundfont_file = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
WORKING_DIR = "./tests/current_test"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _prepare_image(uploaded_file, working_dir):
    """
    Handle PDF-to-image conversion if needed. Returns the path to the image
    that should be processed by the AI engine.
    """
    local_file_path = os.path.join(working_dir, uploaded_file.name)
    with open(local_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.lower().endswith(".pdf"):
        st.info("PDF format detected. Extracting the first page for analysis...")
        pages = convert_from_path(
            local_file_path,
            dpi=300,
            first_page=1,
            last_page=1,
            output_folder=working_dir
        )
        if not pages:
            raise ValueError("The PDF appears to be empty or unreadable.")

        image_to_process = os.path.join(working_dir, "extracted_page_1.png")
        pages[0].save(image_to_process, "PNG")
        return image_to_process
    else:
        return local_file_path


def _clean_working_dir(working_dir):
    """Clear all files in the working directory without removing the dir itself."""
    os.makedirs(working_dir, exist_ok=True)
    for filename in os.listdir(working_dir):
        file_path = os.path.join(working_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def _rescue_oemer_cache(image_path, working_dir):
    """Move oemer's diagnostic images from input dir to working dir."""
    input_dir = os.path.dirname(os.path.abspath(image_path))
    cache_files = ["staffs.png", "noteheads.png", "clefs_keys.png", "stems_rests.png", "bboxes.png"]
    for cache_file in cache_files:
        source_path = os.path.join(input_dir, cache_file)
        if os.path.exists(source_path) and input_dir != os.path.abspath(working_dir):
            shutil.move(source_path, os.path.join(working_dir, cache_file))


def _extract_notes_text(xml_path):
    """Parse a MusicXML file and return a formatted string of detected notes."""
    try:
        score = converter.parse(xml_path)
        extracted_notes = []
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                extracted_notes.append(f"{element.pitch.nameWithOctave} ({element.duration.type})")
            elif isinstance(element, chord.Chord):
                chord_pitches = "-".join(p.nameWithOctave for p in element.pitches)
                extracted_notes.append(f"[{chord_pitches}] ({element.duration.type})")
        return extracted_notes
    except Exception as e:
        return [f"⚠️ Could not parse notes: {e}"]


def _run_single_engine(engine_name, image_path, working_dir, conf=0.25, container=None):
    """
    Run a single AI engine and return a results dict.
    If container is provided, outputs are rendered into that Streamlit container.
    """
    ctx = container or st

    t_start = time.perf_counter()

    if engine_name == "Oemer Baseline":
        xml_result = generation_workflow_oemer(image_path, output_dir=working_dir)
        _rescue_oemer_cache(image_path, working_dir)
    elif engine_name == "Custom YOLO Model":
        xml_result = generation_workflow_custom_yolo(image_path, output_dir=working_dir, conf=conf)
    else:
        raise ValueError(f"Unknown AI engine: {engine_name}")

    t_end = time.perf_counter()
    processing_time = t_end - t_start

    # Synthesize audio
    audio_result = convert_xml_to_mp3(xml_result, soundfont_file)

    # Extract notes
    notes = _extract_notes_text(xml_result)

    # Collect diagnostic images
    diag_images = []
    for f in os.listdir(working_dir):
        if f.endswith(".png") and f not in [uploaded_file.name, "extracted_page_1.png"]:
            diag_images.append(os.path.join(working_dir, f))

    return {
        "engine": engine_name,
        "xml_path": xml_result,
        "audio_path": audio_result,
        "processing_time": processing_time,
        "notes": notes,
        "note_count": len([n for n in notes if not n.startswith("⚠️")]),
        "diag_images": diag_images,
    }


def _display_results(result, container=None):
    """Render the results of a single engine run into a Streamlit container."""
    ctx = container or st

    ctx.metric(
        label="⏱️ AI Processing Time",
        value=f"{result['processing_time']:.2f}s",
        help=f"Time measured exclusively for '{result['engine']}' inference (PDF parsing and audio synthesis excluded)."
    )

    # Audio playback
    ctx.subheader("🔊 Audio Output")
    with open(result["audio_path"], "rb") as af:
        ctx.audio(af.read(), format="audio/wav")

    # Note sequence
    ctx.subheader(f"🎶 Detected Notes ({result['note_count']})")
    with ctx.expander("Show Parsed Notes"):
        note_string = " ➔ ".join(result["notes"])
        ctx.write(note_string)

    # Diagnostic images
    ctx.subheader("🖼️ Visual Diagnostics")
    if result["diag_images"]:
        for diag_path in result["diag_images"]:
            clean_name = os.path.basename(diag_path).replace(".png", "").replace("_", " ").title()
            ctx.image(diag_path, caption=f"AI Layer: {clean_name}", use_container_width=True)
    else:
        ctx.info("No diagnostic images available for this run.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    st.info(f"File uploaded: {uploaded_file.name}")

    # --- A/B TESTING SETUP ---
    col_config1, col_config2 = st.columns([2, 1])

    with col_config1:
        selected_model = st.selectbox(
            "🤖 Select the AI Engine:",
            ["Oemer Baseline", "Custom YOLO Model", "🔬 Benchmark (Both Models)"]
        )

    with col_config2:
        # Confidence slider — only relevant for Custom YOLO
        yolo_conf = st.slider(
            "🎯 YOLO Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Lower = more detections (including noise). Higher = only confident detections."
        )

    if st.button("🚀 Generate Audio", type="primary"):

        # --- CLEAN & PREPARE ---
        _clean_working_dir(WORKING_DIR)

        with st.spinner("Preparing file and analyzing notes... (This may take a while)"):
            try:
                image_to_process = _prepare_image(uploaded_file, WORKING_DIR)
                st.image(image_to_process, caption="Input Image", use_container_width=True)

                # ===================================================================
                # MODE 1 & 2: Single engine (Oemer or Custom YOLO)
                # ===================================================================
                if selected_model in ["Oemer Baseline", "Custom YOLO Model"]:
                    result = _run_single_engine(
                        selected_model, image_to_process, WORKING_DIR, conf=yolo_conf
                    )
                    st.success(f"✅ Conversion Complete with {selected_model}!")
                    _display_results(result)

                # ===================================================================
                # MODE 3: Benchmark — run BOTH engines side-by-side
                # ===================================================================
                elif selected_model == "🔬 Benchmark (Both Models)":
                    st.info("Running both engines for comparison... This will take longer.")

                    # --- Run Oemer first ---
                    oemer_dir = os.path.join(WORKING_DIR, "benchmark_oemer")
                    os.makedirs(oemer_dir, exist_ok=True)

                    # Copy the image to oemer's working directory
                    oemer_image = os.path.join(oemer_dir, os.path.basename(image_to_process))
                    shutil.copy2(image_to_process, oemer_image)

                    oemer_result = _run_single_engine(
                        "Oemer Baseline", oemer_image, oemer_dir
                    )

                    # --- Run Custom YOLO ---
                    yolo_dir = os.path.join(WORKING_DIR, "benchmark_yolo")
                    os.makedirs(yolo_dir, exist_ok=True)

                    yolo_image = os.path.join(yolo_dir, os.path.basename(image_to_process))
                    shutil.copy2(image_to_process, yolo_image)

                    yolo_result = _run_single_engine(
                        "Custom YOLO Model", yolo_image, yolo_dir, conf=yolo_conf
                    )

                    # --- Display side-by-side ---
                    st.success("✅ Benchmark Complete!")

                    # Summary table
                    st.subheader("📊 Benchmark Comparison")
                    st.markdown("---")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric("Metric", "—")

                    with metric_col2:
                        st.metric(
                            "⏱️ Oemer Time",
                            f"{oemer_result['processing_time']:.2f}s"
                        )

                    with metric_col3:
                        # Show delta vs Oemer
                        delta = yolo_result['processing_time'] - oemer_result['processing_time']
                        st.metric(
                            "⏱️ YOLO Time",
                            f"{yolo_result['processing_time']:.2f}s",
                            delta=f"{delta:+.2f}s vs Oemer",
                            delta_color="inverse"  # negative delta = YOLO is faster = good
                        )

                    # Notes count comparison
                    note_col1, note_col2, note_col3 = st.columns(3)
                    with note_col1:
                        st.metric("📝", "Notes Detected")
                    with note_col2:
                        st.metric("Oemer", str(oemer_result['note_count']))
                    with note_col3:
                        note_delta = yolo_result['note_count'] - oemer_result['note_count']
                        st.metric(
                            "YOLO",
                            str(yolo_result['note_count']),
                            delta=f"{note_delta:+d} vs Oemer"
                        )

                    # Confidence threshold display
                    conf_col1, conf_col2, conf_col3 = st.columns(3)
                    with conf_col1:
                        st.metric("🎯", "Confidence")
                    with conf_col2:
                        st.metric("Oemer", "N/A")
                    with conf_col3:
                        st.metric("YOLO", f"{yolo_conf:.2f}")

                    st.markdown("---")

                    # Side-by-side detailed results
                    col_oemer, col_yolo = st.columns(2)

                    with col_oemer:
                        st.header("🅰️ Oemer Baseline")
                        _display_results(oemer_result, container=col_oemer)

                    with col_yolo:
                        st.header("🅱️ Custom YOLO Model")
                        _display_results(yolo_result, container=col_yolo)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                import traceback
                st.code(traceback.format_exc())