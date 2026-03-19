import os
import shutil
import streamlit as st
from pdf2image import convert_from_path
from core.image_processing import generate_musicxml
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
    
    if st.button("Generate Audio"):
        
        # I clean up the current_test folder completely to ensure no old files mix with the new run
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)
        os.makedirs(WORKING_DIR, exist_ok=True)
        
        # I save the newly uploaded file directly into my testing workspace
        local_file_path = os.path.join(WORKING_DIR, uploaded_file.name)
        with open(local_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Preparing file and analyzing notes... (This may take a while)"):
            try:
                # --- NEW PDF HANDLING LOGIC ---
                # I check if the user uploaded a PDF
                if uploaded_file.name.lower().endswith(".pdf"):
                    st.info("PDF format detected. I am extracting the first page for analysis...")
                    
                    # I convert the PDF into high-quality images (300 DPI is great for reading small notes)
                    pages = convert_from_path(local_file_path, dpi=300)
                    
                    if not pages:
                        raise ValueError("The PDF appears to be empty or unreadable.")
                        
                    # Since oemer processes one page at a time, I extract just the first page and save it as a PNG
                    image_to_process = os.path.join(WORKING_DIR, "extracted_page_1.png")
                    pages[0].save(image_to_process, "PNG")
                    
                    # I show the user the extracted page
                    st.image(image_to_process, caption="Extracted Page 1", use_container_width=True)
                else:
                    # If it's already an image, I just use it directly and display it
                    image_to_process = local_file_path
                    st.image(image_to_process, caption="Original Uploaded Image", use_container_width=True)

                # --- OMER PROCESSING ---
                # I extract the XML, passing either the original image or the extracted PDF page
                xml_result = generate_musicxml(image_to_process, output_dir=WORKING_DIR)
                
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