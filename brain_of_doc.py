#setting groq up
import os
#GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
#print(GROQ_API_KEY)
GROQ_API_KEY="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"

#Convert img to base64
import base64

img_path="ringworm.jpeg"
img_file=open(img_path,"rb")
encoded_img=base64.b64encode(img_file.read()).decode("utf-8")

#setup the llm

from groq import Groq

query="What is wrong with my skin"

client = Groq(api_key=GROQ_API_KEY)

# Define model and message
model = "llama-3.2-90b-vision-preview"
query = "What is wrong with my skin? Just give me your diagnosis"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_img}"
                }
            }
        ]
    }
]

# Call API
chat_completion = client.chat.completions.create(
    model=model,
    messages=messages,
)

# Print result
print(chat_completion.choices[0].message.content)




