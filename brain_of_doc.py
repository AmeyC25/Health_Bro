#setting groq up
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print(GROQ_API_KEY)


#Convert img to base64
import base64

#img_path="ringworm.jpeg"


def encode_image(image_path):   
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')

#setup the llm

from groq import Groq

query="What is wrong with my skin"

client = Groq(api_key=GROQ_API_KEY)

# Define model and message

from groq import Groq

query="Is there something wrong with my face?"
model="llama-3.2-90b-vision-preview"

def analyze_image_with_query(query, model, encoded_image):
    client=Groq()  
    messages=[
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
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content