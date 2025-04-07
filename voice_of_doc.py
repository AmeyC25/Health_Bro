from gtts import gTTS
import os
import subprocess
import platform
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Converts text to speech using gTTS and plays it.
    
    Parameters:
    - input_text: Text to convert to speech
    - output_filepath: Path to save the audio file
    
    Returns:
    - Path to the generated audio file
    """
    try:
        language = "en"
        
        audioobj = gTTS(
            text=input_text,
            lang=language,
            slow=False
        )
        audioobj.save(output_filepath)
        logging.info(f"Audio saved to {output_filepath}")
        
        # Play the audio based on the operating system
        os_name = platform.system()
        try:
            if os_name == "Darwin":  # macOS
                subprocess.run(['afplay', output_filepath])
            elif os_name == "Windows":  # Windows
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
            elif os_name == "Linux":  # Linux
                subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
            else:
                raise OSError("Unsupported operating system")
        except Exception as e:
            logging.error(f"An error occurred while trying to play the audio: {e}")
        
        return output_filepath
    except Exception as e:
        logging.error(f"An error occurred in text-to-speech conversion: {e}")
        return None

# Test function when run directly
if __name__ == "__main__":
    input_text = "Hi this is AI with Hassan, autoplay testing!"
    text_to_speech_with_gtts(input_text=input_text, output_filepath="gtts_testing_autoplay.mp3")