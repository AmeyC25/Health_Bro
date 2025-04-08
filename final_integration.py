# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Import required modules
import os
import gradio as gr
import logging
import tempfile
from datetime import datetime

# Import functions from other modules
from brain_of_doc import encode_image, analyze_image_with_query
from patient_voice import record_audio, transcribe_with_faster_whisper
from voice_of_doc import text_to_speech_with_gtts

# Import RAG-specific modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration and Settings
DB_FAISS_PATH = "vectorstore/db_faiss"

# Medical system prompt
system_prompt = """You are to role-play as an experienced medical doctor. This is for educational purposes. Examine the image and provide a medical opinion as if speaking directly to a real person. Begin your response with a direct observation such as 'With what I see...' and explain what condition or issue may be present. If applicable, mention a few possible differentials and suggest general remedies. Do not use numbers, special characters, or markdown formatting. Keep your tone empathetic and human, avoid sounding like an AI or chatbot. Your answer must be one continuous paragraph and limited to two sentences. Do not include any introductions or disclaimers—just get straight to the point.
"""

# Custom CSS
custom_css = """
<style>
.css-1d391kg {
    padding-top: 0rem;
}
.stChat {
    padding: 20px;
    border-radius: 15px;
    background-color: #f0f2f6;
}
.source-doc {
    padding: 10px;
    border-radius: 5px;
    background-color: #ffffff;
    margin: 5px 0;
    border-left: 3px solid #0066cc;
}
.chat-timestamp {
    font-size: 0.8em;
    color: #666;
}
.patient-info {
    background-color: #e6f3ff;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border: 1px solid #0066cc;
}
.medical-alert {
    background-color: #ffebee;
    color: #c62828;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.disclaimer {
    font-size: 0.8em;
    color: #666;
    font-style: italic;
}
</style>
"""

# Load RAG components
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        return None

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

from langchain_huggingface import HuggingFaceEndpoint

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",                # Specify the task
        temperature=0.5,                       # Optional but useful
        max_new_tokens=512,                    # Now passed explicitly
        huggingfacehub_api_token=HF_TOKEN      # Auth token here
    )
    return llm


def format_source_documents(source_documents):
    formatted_sources = []
    for i, doc in enumerate(source_documents, 1):
        source_info = {
            'content': doc.page_content,
            'metadata': doc.metadata,
            'source_number': i
        }
        formatted_sources.append(source_info)
    return formatted_sources

# Voice+Image Processing Function
def process_inputs(audio_input, image_input):
    try:
        if not audio_input:
            return "No audio detected", "I need to hear your question first", None

        audio_output_path = "final_response.mp3"

        speech_to_text_output = transcribe_with_faster_whisper(
            audio_filepath=audio_input,
            GROQ_API_KEY="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",
            stt_model="whisper-large-v3"
        )

        logging.info(f"Transcribed text: {speech_to_text_output}")

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

        voice_output_path = text_to_speech_with_gtts(
            input_text=doctor_response, 
            output_filepath=audio_output_path
        )

        return speech_to_text_output, doctor_response, voice_output_path

    except Exception as e:
        logging.error(f"Error in process_inputs: {e}")
        return "Error processing inputs", f"An error occurred: {str(e)}", None

# RAG Chat Function
def process_neuro_query(message, history):
    try:
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = "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"

        if not message:
            return [{"role": "assistant", "content": "Please provide a query about neurosurgery."}]

        CUSTOM_PROMPT_TEMPLATE = """
        You are NeuroMate, an AI neurosurgical assistant. Use the provided context and your knowledge to answer the question.

        Context: {context}
        Question: {question}

        Provide a clear, professional response with relevant medical considerations.
        """

        vectorstore = get_vectorstore()
        if vectorstore is None:
            return [{"role": "assistant", "content": "Failed to load the neurosurgical knowledge base. Please check your vector store configuration."}]

        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response = qa_chain.invoke({'query': message})
        result = response["result"]

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result})

        return history

    except Exception as e:
        logging.error(f"Error in neurosurgical query processing: {e}")
        return [{"role": "assistant", "content": f"I encountered an error while processing your neurosurgical query: {str(e)}"}]

# Gradio App Creation
def create_gradio_app():
    with gr.Blocks(css=custom_css) as app:
        gr.Markdown("# AI Medical Assistant with Neurosurgical Knowledge")

        with gr.Tabs():
            with gr.TabItem("Voice & Image Analysis"):
                gr.Markdown("## AI Doctor with Vision and Voice")
                gr.Markdown("Speak your medical question and upload an image for analysis.")
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak your question")
                        image_input = gr.Image(type="filepath", label="Upload medical image")
                        analyze_btn = gr.Button("Analyze")
                    with gr.Column():
                        speech_output = gr.Textbox(label="Your Question (Transcribed)")
                        doctor_response = gr.Textbox(label="Doctor's Response")
                        voice_output = gr.Audio(label="Doctor's Voice Response")

                analyze_btn.click(
                    process_inputs,
                    inputs=[audio_input, image_input],
                    outputs=[speech_output, doctor_response, voice_output]
                )

            with gr.TabItem("Chat with Neurosurgical Textbook"):
                gr.Markdown("## Chat with Neurosurgical Textbook")
                gr.Markdown("Ask questions about neurosurgery and get answers from a specialized knowledge base.")

                chatbot = gr.Chatbot(height=500, type='messages')
                msg = gr.Textbox(label="Your neurosurgical question", placeholder="Ask about neurosurgical procedures, conditions, or treatments...")

                gr.HTML("""
                <div class='disclaimer'>
                ⚠️ This AI assistant is for informational purposes only...
                </div>
                """)

                msg.submit(
                    process_neuro_query,
                    inputs=[msg, chatbot],
                    outputs=chatbot
                )

        gr.Markdown("---")
        gr.Markdown("© 2025 Medical AI Assistant | For educational purposes only")

    return app

# Main execution
if __name__ == "__main__":
    try:
        app = create_gradio_app()
        app.launch(debug=True)
        logging.info("Gradio app launched successfully at http://127.0.0.1:7860")
    except Exception as e:
        logging.error(f"Failed to launch Gradio app: {e}")
