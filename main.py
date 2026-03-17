import os
from core.image_processing import generate_musicxml
from core.audio_synthesis import convert_xml_to_mp3

if __name__ == "__main__":
    # I define the input paths for my test image and the soundfont
    test_image = "partitura_test.png" 
    soundfont_file = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    
    # I define the output directory for the current test run
    output_folder = os.path.join("tests", "current_test")
    
    # I create the output directory if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)
    
    # I check if the input image exists before starting the AI engine
    if os.path.exists(test_image):
        print(f"Starting processing for image: {test_image}...")
        
        try:
            # I call the core module and instruct it to save the MusicXML in the test folder
            xml_result = generate_musicxml(test_image, output_dir=output_folder)
            print(f"Success! MusicXML file saved at: {xml_result}")
            
            # I trigger the audio conversion process
            # The .mid and .wav files will automatically be saved in the same folder as the XML
            if os.path.exists(soundfont_file):
                audio_result = convert_xml_to_mp3(xml_result, soundfont_file)
                print(f"Magic complete! Audio file saved at: {audio_result}")
            else:
                print("Error: Cannot find the SoundFont file for audio synthesis.")
                
        except KeyError:
            print("Error: No musical measures found. The sheet music is corrupted or too complex.")
        except Exception as e:
            print(f"Error: An unexpected error occurred: {e}")
    else:
        print(f"Please add an image named '{test_image}' in the root folder.")