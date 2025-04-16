import os
import base64
import logging
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get API key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def encode_image(image_path):
    """
    Encodes an image file to base64.
    
    Parameters:
    - image_path: Path to the image file
    
    Returns:
    - Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None

def analyze_image_with_query(query, encoded_image, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    Analyzes an image with a query using the Groq API.
    
    Parameters:
    - query: Text query to accompany the image
    - encoded_image: Base64 encoded image
    - model: Model to use for analysis
    
    Returns:
    - Analysis result as text
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        return "I couldn't analyze the image. Please try again later."

# Test function when run directly
if __name__ == "__main__":
    if GROQ_API_KEY:
        logging.info("GROQ API key found")
    else:
        logging.error("GROQ API key not found in environment variables")