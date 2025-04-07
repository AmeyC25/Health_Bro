import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from faster_whisper import WhisperModel

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Records audio from the microphone and saves it as an MP3 file.
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Convert audio data to WAV format
            wav_data = BytesIO(audio_data.get_wav_data())
            
            # Load the WAV data with pydub
            audio_segment = AudioSegment.from_wav(wav_data)
            
            # Export as MP3
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")
            return True
    except sr.WaitTimeoutError:
        logging.error("Timeout occurred while waiting for voice input")
        return False
    except Exception as e:
        logging.error(f"An error occurred while recording: {e}")
        return False

def transcribe_with_faster_whisper(audio_filepath, GROQ_API_KEY=None, stt_model=None):
    """
    Transcribes the audio file using the faster-whisper model.
    
    Parameters:
    - audio_filepath: Path to the audio file
    - GROQ_API_KEY: Optional API key (not used by faster-whisper but needed for interface compatibility)
    - stt_model: Optional model name (defaults to "base" if not specified)
    """
    try:
        # Use specified model or default to "base"
        model_name = "base" if stt_model is None else "base"
        
        # Load model (downloaded the first time)
        model = WhisperModel(model_name, device="cpu", compute_type="float32")
        
        # Transcribe audio
        segments, info = model.transcribe(audio_filepath, beam_size=5)
        
        # Collect all segments into a single transcript
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        
        logging.info(f"Transcription completed: {len(transcription)} characters")
        return transcription.strip()
    
    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")
        return "I couldn't understand what you said. Please try again."

# Main flow
if __name__ == "__main__":
    audio_filepath = "patient_voice_test.mp3"
    
    # Step 1: Record audio
    recording_success = record_audio(file_path=audio_filepath, timeout=20, phrase_time_limit=10)
    
    # Step 2: Transcribe locally
    if recording_success:
        transcription = transcribe_with_faster_whisper(audio_filepath)
        
        if transcription:
            print("\n--- Transcription ---")
            print(transcription)
        else:
            print("\nTranscription failed. Check logs for details.")
    else:
        print("\nRecording failed. Check logs for details.")