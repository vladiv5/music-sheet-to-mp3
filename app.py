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
st.set_page_config(page_title="Music Sheet to MP3", page_icon="🎵", layout="centered")

st.title("🎵 OMR Audio Synthesizer")
st.markdown("Upload a clean image or PDF of a music sheet and let the AI convert it into a playable audio file.")

# I updated the uploader to accept PDF files as well
uploaded_file = st.file_uploader("Upload your sheet music (PDF/PNG/JPG)", type=["png", "jpg", "jpeg", "pdf"])

soundfont_file = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
WORKING_DIR = "./tests/current_test"

if uploaded_file is not None:
    # I display the uploaded file name
    st.info(f"File uploaded: {uploaded_file.name}")
    
    # --- A/B TESTING SETUP ---
    # I add a dropdown to let the user select which AI model to use
    selected_model = st.selectbox(
        "Select the AI Engine:",
        ["Oemer Baseline", "Custom CV Model"]
    )
    
    if st.button("Generate Audio"):
        
        # I ensure the directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)
        
        # I empty the directory by deleting the files inside it, rather than destroying the directory itself
        # This prevents Docker from crashing with a "Device or resource busy" error on the mounted volume
        for filename in os.listdir(WORKING_DIR):
            file_path = os.path.join(WORKING_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        
        # I save the newly uploaded file directly into my testing workspace
        local_file_path = os.path.join(WORKING_DIR, uploaded_file.name)
        with open(local_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Preparing file and analyzing notes... (This may take a while)"):
            try:
                # --- NEW PDF HANDLING LOGIC ---
                # I check if the user uploaded a PDF
                if uploaded_file.name.lower().endswith(".pdf"):
                    st.info("PDF format detected. I am extracting strictly the first page for analysis...")
                    
                    # I convert only the first page and ensure intermediate massive PPM files 
                    # go straight to WORKING_DIR (mapped later to D:) instead of the root /tmp
                    pages = convert_from_path(
                        local_file_path, 
                        dpi=300, 
                        first_page=1, 
                        last_page=1,
                        output_folder=WORKING_DIR
                    )
                    
                    if not pages:
                        raise ValueError("The PDF appears to be empty or unreadable.")
                        
                    # I extract just the first page and save it as a PNG
                    image_to_process = os.path.join(WORKING_DIR, "extracted_page_1.png")
                    pages[0].save(image_to_process, "PNG")
                    
                    # I show the user the extracted page
                    st.image(image_to_process, caption="Extracted Page 1", use_container_width=True)
                else:
                    # If it's already an image, I just use it directly and display it
                    image_to_process = local_file_path
                    st.image(image_to_process, caption="Original Uploaded Image", use_container_width=True)

                # --- AI PROCESSING WITH BENCHMARKING ---
                # The timer wraps ONLY the AI inference step, regardless of which engine is selected.
                # This ensures A/B comparisons between Oemer and any future custom model are fair.
                t_start = time.perf_counter()

                if selected_model == "Oemer Baseline":
                    xml_result = generation_workflow_oemer(image_to_process, output_dir=WORKING_DIR)
                elif selected_model == "Custom CV Model":
                    xml_result = generation_workflow_custom_yolo(image_to_process, output_dir=WORKING_DIR)
                else:
                    raise ValueError(f"Unknown AI engine selected: {selected_model}")

                t_end = time.perf_counter()
                processing_time = t_end - t_start
                
                # I catch oemer's rogue diagnostic images and move them to my working directory
                input_dir = os.path.dirname(os.path.abspath(image_to_process))
                cache_files = ["staffs.png", "noteheads.png", "clefs_keys.png", "stems_rests.png", "bboxes.png"]
                
                for cache_file in cache_files:
                    source_path = os.path.join(input_dir, cache_file)
                    if os.path.exists(source_path) and input_dir != os.path.abspath(WORKING_DIR):
                        shutil.move(source_path, os.path.join(WORKING_DIR, cache_file))
                
                st.info("MusicXML successfully generated. Synthesizing audio...")
                
                # I synthesize the audio
                audio_result = convert_xml_to_mp3(xml_result, soundfont_file)
                
                st.success("Conversion Complete!")
                
                # I read the audio file bytes so I can play it directly in the browser
                with open(audio_result, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    
                st.audio(audio_bytes, format="audio/wav")

                # --- BENCHMARKING RESULT ---
                st.metric(
                    label="⏱️ AI Processing Time",
                    value=f"{processing_time:.2f}s",
                    help=f"Time measured exclusively for the '{selected_model}' inference step (PDF parsing and audio synthesis excluded). Use this to A/B compare engines."
                )

                # --- NOTE SEQUENCE EXTRACTION ---
                st.subheader("🎶 Detected Note Sequence")
                st.markdown("Here is the exact sequence of notes the AI extracted before converting to audio:")
                
                # I use a collapsible expander so it doesn't clutter the whole page
                with st.expander("Show Parsed Notes"):
                    try:
                        # I load the generated MusicXML file using music21
                        score = converter.parse(xml_result)
                        extracted_notes = []
                        
                        # I flatten the score to easily loop through all musical elements in order
                        for element in score.flatten().notes:
                            if isinstance(element, note.Note):
                                # If it's a single note, I grab its name, octave, and duration
                                extracted_notes.append(f"{element.pitch.nameWithOctave} ({element.duration.type})")
                            elif isinstance(element, chord.Chord):
                                # If it's a chord (multiple notes stacked), I extract all pitches
                                chord_pitches = "-".join(p.nameWithOctave for p in element.pitches)
                                extracted_notes.append(f"[{chord_pitches}] ({element.duration.type})")
                                
                        # I join the notes with an arrow for a clean visual timeline
                        note_string = " ➔ ".join(extracted_notes)
                        st.write(note_string)
                        
                    except Exception as e:
                        st.warning(f"I could not extract the text notes. Reason: {e}")

                # --- VISUAL DIAGNOSTICS SECTION ---
                st.subheader("Visual Diagnostics")
                st.markdown("Here is how the AI detected the elements on your sheet music:")
                
                # I look directly in my testing workspace for the diagnostic images
                diagnostic_images = [
                    f for f in os.listdir(WORKING_DIR) 
                    if f.endswith(".png") and f not in [uploaded_file.name, "extracted_page_1.png"]
                ]
                
                if diagnostic_images:
                    for diag_img in diagnostic_images:
                        diag_path = os.path.join(WORKING_DIR, diag_img)
                        clean_name = diag_img.replace(".png", "").replace("_", " ").title()
                        st.image(diag_path, caption=f"AI Detection Layer: {clean_name}", use_container_width=True)
                else:
                    st.info("No diagnostic overlay image was found for this run.")
                    
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")