import os
import shutil
import time
import streamlit as st
from music21 import converter, note, chord
from pdf2image import convert_from_path
from core.image_processing import generation_workflow_oemer
from models.custom_yolo_inference import generation_workflow_custom_yolo
from models.primitive_yolo_inference import generation_workflow_primitive_yolo
from core.audio_synthesis import convert_xml_to_mp3
from core.density_scorer import compute_density_score

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

def _prepare_images(uploaded_file, working_dir):
    """
    Handle PDF-to-image conversion if needed. Returns a list of paths to the images
    that should be processed by the AI engine.
    """
    local_file_path = os.path.join(working_dir, uploaded_file.name)
    with open(local_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.lower().endswith(".pdf"):
        st.info("PDF format detected. Extracting all pages for analysis...")
        pages = convert_from_path(
            local_file_path,
            dpi=300,
            output_folder=working_dir
        )
        if not pages:
            raise ValueError("The PDF appears to be empty or unreadable.")

        image_paths = []
        for idx, page in enumerate(pages):
            image_to_process = os.path.join(working_dir, f"extracted_page_{idx+1}.png")
            page.save(image_to_process, "PNG")
            image_paths.append(image_to_process)
        return image_paths
    else:
        return [local_file_path]


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


def _run_single_engine(engine_name, image_paths, working_dir, conf=0.25, iou=0.7,
                       dx_tolerance=15.0, enable_beam_correction=True,
                       use_sahi=False, sahi_slice_size=640, sahi_overlap=0.25,
                       staves_per_system=1, time_signature="4/4", instrument="Acoustic Grand Piano",
                       use_system_sahi=False, sahi_systems_per_slice=1, use_ai_barlines=False,
                       octave_shift=0, container=None):
    """
    Run a single AI engine and return a results dict.
    If container is provided, outputs are rendered into that Streamlit container.
    """
    ctx = container or st

    if not isinstance(image_paths, list):
        image_paths = [image_paths]

    t_start = time.perf_counter()

    combined_score = None  # kept for YOLO engines that still use it
    page_xml_paths = []    # used by Oemer's raw XML merge strategy
    final_xml_path = os.path.join(working_dir, "combined_output.musicxml")

    # 1. INIȚIALIZĂM MEMORIA ARMURII AICI:
    current_fifths = None

    # Progress bar for pages
    progress_bar = ctx.progress(0.0, text=f"Processing page 1 of {len(image_paths)}...")

    for idx, img_path in enumerate(image_paths):
        progress_bar.progress((idx) / len(image_paths), text=f"Processing page {idx + 1} of {len(image_paths)} with {engine_name}...")
        
        if engine_name == "Oemer Baseline":
            xml_result = generation_workflow_oemer(img_path, output_dir=working_dir)
            if idx == 0: _rescue_oemer_cache(img_path, working_dir)
            
        elif engine_name == "Custom YOLO Model":
            xml_result = generation_workflow_custom_yolo(
                img_path,
                output_dir=working_dir,
                conf=conf, iou=iou,
                dx_tolerance=dx_tolerance, enable_beam_correction=enable_beam_correction,
                use_sahi=use_sahi, sahi_slice_size=sahi_slice_size, sahi_overlap=sahi_overlap,
            )
            
        elif engine_name == "YOLOv8s Primitives":
            # 2. PRELUĂM ȘI TRIMITEM ARMURA MAI DEPARTE AICI:
            xml_result, current_fifths = generation_workflow_primitive_yolo(
                img_path,
                output_dir=working_dir,
                conf=conf, iou=iou,
                dx_tolerance=dx_tolerance, enable_beam_correction=enable_beam_correction,
                use_sahi=use_sahi, sahi_slice_size=sahi_slice_size, sahi_overlap=sahi_overlap,
                staves_per_system=staves_per_system, time_signature=time_signature,
                use_ai_barlines=use_ai_barlines,
                use_system_sahi=use_system_sahi,
                sahi_systems_per_slice=sahi_systems_per_slice,
                octave_shift=octave_shift,
                inherited_fifths=current_fifths  # <--- Transmitem memoria de la pagina anterioară!
            )
        else:
            raise ValueError(f"Unknown AI engine: {engine_name}")

        # Collect raw XML paths — we will merge at the XML level to avoid
        # music21's internal makeRests/makeTies crashing on Oemer's output.
        page_xml_paths.append(xml_result)

    progress_bar.progress(1.0, text="Stitching complete! Synthesizing audio...")

    # --- XML MERGE STRATEGY ---
    # If only one page, use the file directly without touching it.
    # If multiple pages, merge the <measure> blocks at the raw XML level.
    if len(page_xml_paths) == 1:
        final_xml_path = page_xml_paths[0]
    else:
        try:
            import xml.etree.ElementTree as ET
            ET.register_namespace('', 'http://www.musicxml.org/ns/mxl')

            # Parse the first page as the base document
            base_tree = ET.parse(page_xml_paths[0])
            base_root = base_tree.getroot()

            # Helper: find all <part> elements regardless of namespace
            def find_parts(root):
                parts = root.findall('.//{http://www.musicxml.org/ns/mxl}part')
                if not parts:
                    parts = root.findall('.//part')
                return parts

            base_parts = find_parts(base_root)

            # Track the last measure number per part
            def last_measure_num(part_el):
                measures = [c for c in part_el if c.tag.endswith('measure') or c.tag == 'measure']
                if not measures:
                    return 0
                num = measures[-1].get('number', '0')
                try:
                    return int(num)
                except ValueError:
                    return len(measures)

            for extra_xml in page_xml_paths[1:]:
                extra_tree = ET.parse(extra_xml)
                extra_root = extra_tree.getroot()
                extra_parts = find_parts(extra_root)

                for i, base_part in enumerate(base_parts):
                    if i >= len(extra_parts):
                        break
                    offset = last_measure_num(base_part)
                    extra_part = extra_parts[i]
                    for measure_el in list(extra_part):
                        if measure_el.tag.endswith('measure') or measure_el.tag == 'measure':
                            old_num = int(measure_el.get('number', '1'))
                            measure_el.set('number', str(offset + old_num))
                            base_part.append(measure_el)

            base_tree.write(final_xml_path, xml_declaration=True, encoding='UTF-8')
        except Exception as merge_err:
            print(f"XML merge failed: {merge_err}, falling back to first page only.")
            import shutil as _shutil
            _shutil.copy2(page_xml_paths[0], final_xml_path)


    t_end = time.perf_counter()
    processing_time = t_end - t_start

    # Synthesize audio with instrument selection
    audio_result = convert_xml_to_mp3(final_xml_path, soundfont_file, instrument_name=instrument)

    # Extract notes from the final combined output
    notes = _extract_notes_text(final_xml_path)

    # Collect diagnostic images (sorted by page number)
    diag_images = []
    excluded = {uploaded_file.name}
    # Exclude all extracted_page_*.png files from diagnostics
    for f in os.listdir(working_dir):
        if f.startswith("extracted_page_") and f.endswith(".png"):
            excluded.add(f)
    for f in sorted(os.listdir(working_dir)):
        if f.endswith(".png") and f not in excluded:
            diag_images.append(os.path.join(working_dir, f))

    return {
        "engine": engine_name,
        "xml_path": final_xml_path,
        "midi_path": final_xml_path.replace(".musicxml", ".mid"),
        "audio_path": audio_result,
        "processing_time": processing_time,
        "notes": notes,
        "note_count": len([n for n in notes if not n.startswith("⚠️")]),
        "chord_count": len([n for n in notes if "[" in n]),
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

    # Download Buttons
    ctx.subheader("💾 Export Options")
    col1, col2 = ctx.columns(2)
    with open(result["xml_path"], "r", encoding="utf-8") as f:
        col1.download_button(
            label="Download MusicXML",
            data=f.read(),
            file_name="output.musicxml",
            mime="application/vnd.recordare.musicxml+xml"
        )
    if "midi_path" in result and os.path.exists(result["midi_path"]):
        with open(result["midi_path"], "rb") as f:
            col2.download_button(
                label="Download MIDI",
                data=f.read(),
                file_name="output.mid",
                mime="audio/midi"
            )

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
# Developer Panel (Sidebar)
# ---------------------------------------------------------------------------

st.sidebar.title("🛠️ Developer Panel")
st.sidebar.caption("Fine-tune parameters for Custom YOLO engine.")

with st.sidebar.expander("🎯 YOLO Detection", expanded=True):
    yolo_conf = st.slider(
        "Confidence Threshold",
        min_value=0.05, max_value=0.95, value=0.30, step=0.05,
        help="Lower = more detections (including noise). Higher = only confident detections."
    )
    yolo_iou = st.slider(
        "NMS IoU Threshold",
        min_value=0.1, max_value=0.9, value=0.8, step=0.1,
        help="Higher = allows overlapping noteheads (crucial for chords)."
    )

with st.sidebar.expander("🎹 Polyphony & Rhythm", expanded=True):
    yolo_dx = st.slider(
        "Chord Grouping Tolerance (px)",
        min_value=5.0, max_value=30.0, value=11.0, step=1.0,
        help="Max X-distance between notes to be grouped as a chord."
    )
    staves_per_system = st.slider(
        "Staves per System",
        min_value=1, max_value=4, value=1, step=1,
        help="1 = Solo Instrument / Single Staff, 2 = Piano (Left/Right hand played parallel)"
    )
    time_signature = st.selectbox(
        "Time Signature",
        ["4/4", "3/4", "2/4", "2/2", "3/8", "5/4", "5/8", "6/4", "6/8", "7/8", "9/8", "12/8"],
        index=0,
        help="Matches the score's time signature. Menuet = 3/4, Unravel = 4/4, waltzes = 3/4, etc."
    )
    enable_beams = st.checkbox(
        "Enable Geometric Beam Analysis",
        value=True,
        help="Overrides YOLO's rhythm by analyzing stems for beams/flags (HoughLinesP)."
    )
    octave_shift = st.slider(
        "Global Octave Shift",
        min_value=-2, max_value=2, value=0, step=1,
        help="Shift the entire sheet music up or down by octaves (e.g., 8va = +1)."
    )

with st.sidebar.expander("🧠 Smart Auto / SAHI", expanded=True):
    density_threshold = st.slider(
        "Density Threshold",
        min_value=0.10, max_value=0.80, value=0.35, step=0.05,
        help="Score above which SAHI is activated automatically in Smart Auto mode."
    )
    force_sahi = st.checkbox(
        "⚡ Force SAHI (Debug)",
        value=False,
        help="Override Smart Auto: always use SAHI sliced inference regardless of density."
    )
    use_system_sahi = st.checkbox(
        "✂️ Enable System-SAHI",
        value=False,
        help="Slices the page horizontally by groups of staves to improve recall on dense sheets."
    )
    sahi_systems_per_slice = st.slider(
        "Systems per SAHI Slice",
        min_value=1, max_value=4, value=1, step=1,
        help="How many complete musical systems to group together per horizontal slice."
    )
with st.sidebar.expander("🛠️ Advanced Settings", expanded=False):
    audio_instrument = st.selectbox(
        "Audio Instrument",
        ["Acoustic Grand Piano", "Violin", "Flute", "Acoustic Guitar", "Trumpet", "Cello"],
        index=0,
        help="Select the instrument for audio synthesis."
    )
    process_first_page_only = st.checkbox(
        "⚡ Fast Mode: Process 1st Page Only",
        value=True,
        help="If checked, ignores the rest of the PDF pages to speed up debugging."
    )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    st.info(f"File uploaded: {uploaded_file.name}")

    # --- Engine Selection ---
    selected_model = st.selectbox(
        "🤖 Select the AI Engine:",
        [
            "YOLOv8s Primitives + AI Barlines (Recommended)",
            "YOLOv8s Primitives (OpenCV Barlines)",
            "Oemer Baseline",
            "🔬 Benchmark (Both Models)",
        ]
    )
    
    if "Oemer" in selected_model:
        st.warning("⚠️ **Note on Memory:** Oemer is high-precision but very memory-intensive. I have implemented automatic image downscaling to prevent system crashes, but it may still take several minutes to process.")

    if st.button("🚀 Generate Audio", type="primary"):

        # --- CLEAN & PREPARE ---
        _clean_working_dir(WORKING_DIR)

        with st.spinner("Preparing file and analyzing notes... (This may take a while)"):
            try:
                image_paths = _prepare_images(uploaded_file, WORKING_DIR)
                if len(image_paths) == 1:
                    st.image(image_paths[0], caption="Input Image", use_container_width=True)
                else:
                    st.info(f"Loaded {len(image_paths)} pages. Previewing the first page...")
                    st.image(image_paths[0], caption="Input Image (Page 1)", use_container_width=True)

                # -------------------------------------------------------------------
                # Fast Mode slicing
                # -------------------------------------------------------------------
                if process_first_page_only and len(image_paths) > 1:
                    image_paths = image_paths[:1]
                    st.info("⚡ Fast Mode active: Processing only the first page of the PDF.")

                yolo_kwargs = dict(
                    conf=yolo_conf, iou=yolo_iou,
                    dx_tolerance=yolo_dx, enable_beam_correction=enable_beams,
                    use_sahi=False, sahi_slice_size=640,
                    sahi_overlap=0.25,
                )
                
                # Primitive model accepts staves_per_system and time_signature
                primitive_kwargs = dict(yolo_kwargs)
                primitive_kwargs["staves_per_system"] = staves_per_system
                primitive_kwargs["time_signature"] = time_signature
                primitive_kwargs["use_system_sahi"] = use_system_sahi
                primitive_kwargs["sahi_systems_per_slice"] = sahi_systems_per_slice
                primitive_kwargs["octave_shift"] = octave_shift

                # ===================================================================
                # MODE 0: Smart Auto — density scorer chooses engine
                # ===================================================================
                if selected_model == "🧠 Smart Auto (Density-Based)":
                    with st.spinner(f"🧮 Analyzing page density for {len(image_paths)} pages..."):
                        density_report = compute_density_score(
                            image_paths[0], # Evaluate density on first page
                            density_threshold=density_threshold,
                        )

                    # --- Display density gauge ---
                    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                    d_col1.metric("Density Score (Pg 1)", f"{density_report.overall_score:.2f}")
                    d_col2.metric("Ink", f"{density_report.ink_density:.2f}")
                    d_col3.metric("Vertical", f"{density_report.vertical_complexity:.2f}")
                    d_col4.metric("Staff Load", f"{density_report.staff_utilization:.2f}")

                    # --- Determine engine and SAHI ---
                    if density_report.recommendation == 'oemer':
                        auto_engine = "Oemer Baseline"
                        auto_sahi = False
                        st.warning(f"{density_report.label} — Routing to **Oemer** "
                                   f"(density {density_report.overall_score:.2f} ≥ "
                                   f"{density_threshold + 0.20:.2f})")
                    elif density_report.recommendation == 'sahi':
                        auto_engine = "Custom YOLO Model"
                        auto_sahi = True
                        st.info(f"{density_report.label} — Routing to **YOLO + SAHI** "
                                f"(density {density_report.overall_score:.2f} ≥ "
                                f"{density_threshold:.2f})")
                    else:
                        auto_engine = "Custom YOLO Model"
                        auto_sahi = force_sahi
                        st.success(f"{density_report.label} — Routing to **YOLO Standard** "
                                   f"(density {density_report.overall_score:.2f} < "
                                   f"{density_threshold:.2f})")

                    yolo_kwargs['use_sahi'] = auto_sahi or force_sahi

                    result = _run_single_engine(
                        auto_engine, image_paths, WORKING_DIR, instrument=audio_instrument, **yolo_kwargs
                    )
                    st.success(f"✅ Conversion Complete with {auto_engine}!")
                    _display_results(result)

                # ===================================================================
                # MODE 1: Explicit Custom YOLO
                # ===================================================================
                if selected_model in ["YOLOv8s Primitives + AI Barlines (Recommended)", "YOLOv8s Primitives (OpenCV Barlines)"]:
                    use_ai_barlines = selected_model == "YOLOv8s Primitives + AI Barlines (Recommended)"
                    primitive_kwargs["use_ai_barlines"] = use_ai_barlines
                    result = _run_single_engine(
                        "YOLOv8s Primitives", image_paths, WORKING_DIR, instrument=audio_instrument, **primitive_kwargs
                    )
                    st.success(f"✅ Conversion Complete with {selected_model}!")
                    _display_results(result)

                elif selected_model in ["Oemer Baseline", "Custom YOLO Model"]:
                    result = _run_single_engine(
                        selected_model, image_paths, WORKING_DIR, instrument=audio_instrument, **yolo_kwargs
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
                        "Custom YOLO Model", yolo_image, yolo_dir, **yolo_kwargs
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
                        st.metric("⏱️ Oemer Time",
                                  f"{oemer_result['processing_time']:.2f}s")
                    with metric_col3:
                        delta = yolo_result['processing_time'] - oemer_result['processing_time']
                        st.metric("⏱️ YOLO Time",
                                  f"{yolo_result['processing_time']:.2f}s",
                                  delta=f"{delta:+.2f}s vs Oemer",
                                  delta_color="inverse")

                    note_col1, note_col2, note_col3 = st.columns(3)
                    with note_col1:
                        st.metric("📝", "Notes Detected")
                    with note_col2:
                        st.metric("Oemer", str(oemer_result['note_count']))
                    with note_col3:
                        note_delta = yolo_result['note_count'] - oemer_result['note_count']
                        st.metric("YOLO", str(yolo_result['note_count']),
                                  delta=f"{note_delta:+d} vs Oemer")

                    chord_col1, chord_col2, chord_col3 = st.columns(3)
                    with chord_col1:
                        st.metric("🎹", "Chords Detected")
                    with chord_col2:
                        st.metric("Oemer", str(oemer_result['chord_count']))
                    with chord_col3:
                        chord_delta = yolo_result['chord_count'] - oemer_result['chord_count']
                        st.metric("YOLO", str(yolo_result['chord_count']),
                                  delta=f"{chord_delta:+d} vs Oemer")

                    conf_col1, conf_col2, conf_col3 = st.columns(3)
                    with conf_col1:
                        st.metric("🎯", "Confidence")
                    with conf_col2:
                        st.metric("Oemer", "N/A")
                    with conf_col3:
                        st.metric("YOLO", f"{yolo_conf:.2f}")

                    st.markdown("---")

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