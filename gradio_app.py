# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#VoiceBot UI with Gradio
import os
import gradio as gr

from brain_of_doc import encode_image, analyze_image_with_query
from patient_voice import record_audio, transcribe_with_groq
from voice_of_doc import text_to_speech_with_gtts

#load_dotenv()

system_prompt="""You are to role-play as an experienced medical doctor. This is for educational purposes. Examine the image and provide a medical opinion as if speaking directly to a real person. Begin your response with a direct observation such as 'With what I see...' and explain what condition or issue may be present. If applicable, mention a few possible differentials and suggest general remedies. Do not use numbers, special characters, or markdown formatting. Keep your tone empathetic and human, avoid sounding like an AI or chatbot. Your answer must be one continuous paragraph and limited to two sentences. Do not include any introductions or disclaimers—just get straight to the point.
"""


def process_inputs(audio_filepath, image_filepath):
    speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                                                 audio_filepath=audio_filepath,
                                                 stt_model="whisper-large-v3")

    # Handle the image input
    if image_filepath:
        doctor_response = analyze_image_with_query(query=system_prompt+speech_to_text_output, encoded_image=encode_image(image_filepath), model="llama-3.2-11b-vision-preview")
    else:
        doctor_response = "No image provided for me to analyze"

    voice_of_doctor = text_to_speech_with_gtts(input_text=doctor_response, output_filepath="final.mp3") 

    return speech_to_text_output, doctor_response, voice_of_doctor


# Create the interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio("Temp.mp3")
    ],
    title="AI Doctor with Vision and Voice"
)

iface.launch(debug=True)

#http://127.0.0.1:7860