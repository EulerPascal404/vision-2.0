from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64, httpx
from langchain_core.retrievers import BaseRetriever, RetrieverOutput
from langchain_core.runnables import Runnable, RunnablePassthrough
from PIL import Image
import base64
import io


import getpass
import os

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/gemini_endpoint', methods=['POST'])
def fn():

    data = request.get_json()
    name = data.get('name')
    age = data.get('age')
    print(f'Received name: {name}, age: {age}')

    base64_image = base64.b64decode(age)
    filename = "hotel_test.jpg"
    with open(filename, 'wb') as f:
        f.write(base64_image)
    
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "AIzaSyA9Ddx0pD-qN9rzbCVaMQ1eNAW5BeIXTqw"

    # Initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    def encode_image(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    image = Image.open('hotel_test.jpg')

    # Download and encode the image
    image_data = encode_image(image)

    # Create a message with the image

    message = HumanMessage(
        content=[
            {"type": "text", "text": "First describe the scene. Then, describe directions for how you would get from one side of the room to the other, and point out key obstacles. Specify the exact depth and distance between objects. Assume you were speaking to a blind person and helping them navigate."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )
    # Invoke the model with the message
    response = model.invoke([message])

    # Print the model's response
    print(response.content)
    return response.content
