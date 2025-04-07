# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import required modules
import os
import gradio as gr
import logging
import tempfile

# Import functions from other modules
from brain_of_doc import encode_image, analyze_image_with_query
from patient_voice import record_audio, transcribe_with_faster_whisper
from voice_of_doc import text_to_speech_with_gtts

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Medical system prompt
system_prompt = """You are to role-play as an experienced medical doctor. This is for educational purposes. Examine the image and provide a medical opinion as if speaking directly to a real person. Begin your response with a direct observation such as 'With what I see...' and explain what condition or issue may be present. If applicable, mention a few possible differentials and suggest general remedies. Do not use numbers, special characters, or markdown formatting. Keep your tone empathetic and human, avoid sounding like an AI or chatbot. Your answer must be one continuous paragraph and limited to two sentences. Do not include any introductions or disclaimers—just get straight to the point.
"""

def process_inputs(audio_input, image_input):
    """
    Process audio and image inputs to generate a medical response.
    
    Parameters:
    - audio_input: Audio file path from Gradio
    - image_input: Image file path from Gradio
    
    Returns:
    - Transcribed text from audio
    - Doctor's response text
    - Path to the doctor's response as audio
    """
    try:
        # Check if we got valid inputs
        if not audio_input:
            return "No audio detected", "I need to hear your question first", None
        
        # Create temporary file paths for outputs
        audio_output_path = "final_response.mp3"
            
        # Step 1: Transcribe patient's speech
        speech_to_text_output = transcribe_with_faster_whisper(
            audio_filepath=audio_input,
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            stt_model="whisper-large-v3"  # This parameter is ignored but kept for compatibility
        )
        
        logging.info(f"Transcribed text: {speech_to_text_output}")
        
        # Step 2: Handle the image analysis
        if image_input:
            try:
                encoded_image = encode_image(image_input)
                if encoded_image:
                    full_query = f"{system_prompt}\nPatient says: {speech_to_text_output}"
                    doctor_response = analyze_image_with_query(
                        query=full_query, 
                        encoded_image=encoded_image, 
                        model="llama-3.2-11b-vision-preview"
                    )
                else:
                    doctor_response = "I couldn't process the image you provided."
            except Exception as e:
                logging.error(f"Error in image analysis: {e}")
                doctor_response = "I'm having trouble analyzing the image. Could you please try again?"
        else:
            doctor_response = "I need an image to examine in order to provide a medical opinion."
        
        # Step 3: Convert the doctor's response to speech
        voice_output_path = text_to_speech_with_gtts(
            input_text=doctor_response, 
            output_filepath=audio_output_path
        )
        
        return speech_to_text_output, doctor_response, voice_output_path
        
    except Exception as e:
        logging.error(f"Error in process_inputs: {e}")
        return "Error processing inputs", f"An error occurred: {str(e)}", None

# Create the interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Speak your question"),
        gr.Image(type="filepath", label="Upload medical image")
    ],
    outputs=[
        gr.Textbox(label="Your Question (Transcribed)"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response")
    ],
    title="AlloEasy",
    description="Speak your medical question and upload an image for analysis. The AI doctor will provide a brief assessment."
)

# Launch the app
if __name__ == "__main__":
    try:
        # Check if GROQ API key is available
        if not os.environ.get("GROQ_API_KEY"):
            logging.warning("GROQ_API_KEY not found in environment variables. The app might not function correctly.")
            
        iface.launch(debug=True)
        logging.info("Gradio app launched successfully at http://127.0.0.1:7860")
    except Exception as e:
        logging.error(f"Failed to launch Gradio app: {e}")