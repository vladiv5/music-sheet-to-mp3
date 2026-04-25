import os
import music21
from midi2audio import FluidSynth

def convert_xml_to_mp3(xml_path: str, soundfont_path: str, instrument_name: str = "Acoustic Grand Piano") -> str:
    # I define the output filenames based on the input xml
    base_name = os.path.splitext(xml_path)[0]
    midi_path = f"{base_name}.mid"
    mp3_path = f"{base_name}.wav" 
    
    # I parse the MusicXML file and convert it to MIDI format
    print("Converting MusicXML to MIDI...")
    parsed_score = music21.converter.parse(xml_path)
    
    # Map selection to music21 instrument objects
    inst_map = {
        "Acoustic Grand Piano": music21.instrument.Piano(),
        "Violin": music21.instrument.Violin(),
        "Cello": music21.instrument.Violoncello(),
        "Flute": music21.instrument.Flute(),
        "Acoustic Guitar": music21.instrument.AcousticGuitar(),
        "Trumpet": music21.instrument.Trumpet()
    }
    
    selected_inst = inst_map.get(instrument_name, music21.instrument.Piano())
    
    # Inject the instrument program change into every part of the score
    if hasattr(parsed_score, 'parts'):
        for part in parsed_score.parts:
            part.insert(0, selected_inst)
            
    parsed_score.write('midi', fp=midi_path)
    
    # I use FluidSynth with a SoundFont to render the MIDI into an audio file
    print("Synthesizing MIDI to audio...")
    fs = FluidSynth(sound_font=soundfont_path)
    fs.midi_to_audio(midi_path, mp3_path)
    
    return mp3_path