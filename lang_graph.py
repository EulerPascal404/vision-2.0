import langgraph as lg
import torch
import tarfile
import os
import requests
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import clip
import requests
from requests.exceptions import ConnectionError
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64, io
from PIL import Image
import base64
from langchain.schema import HumanMessage
from langchain.llms import GooglePalm
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64, httpx
from langchain_core.retrievers import BaseRetriever, RetrieverOutput
from langchain_core.runnables import Runnable, RunnablePassthrough
from PIL import Image
import base64
import io
import json


# Load environment variables
load_dotenv()

# Your Gemini API key from .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Define the ObjectDetectionNode
class ObjectDetectionNode:
    def run(self, image):
        # Load YOLOv5 model for object detection
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        results = model(image)
        objects = results.pandas().xyxy[0]  # bounding boxes and labels
        return objects[['name', 'xmin', 'ymin', 'xmax', 'ymax']]

# Define the DepthEstimationNode with the fix
import torch
import torchvision.transforms as transforms
from PIL import Image

class DepthEstimationNode:
    def run(self, image_path):
        # Load MiDaS model for depth estimation
        model_type = "DPT_Large"  # or "MiDaS_small" for a smaller model
        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Image transformation pipeline (manual conversion to tensor)
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Resize to match MiDaS input requirements
            transforms.ToTensor(),         # Convert to tensor (automatically normalizes to [0, 1])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
        ])

        # Open the image and apply the transform
        image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
        input_image = transform(image).unsqueeze(0)    # Add batch dimension (1, 3, H, W)
        
        # Perform depth estimation
        with torch.no_grad():
            prediction = midas(input_image)  # Model expects input in (1, 3, H, W)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
            ).squeeze()  # Reshape to match the original image size

        # Convert the prediction to a NumPy array
        depth_map = prediction.cpu().numpy()
        return depth_map

# Define the SceneAnalysisNode
class SceneAnalysisNode:
    def run(self, image_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(["a street", "a park", "a building", "a house", "trees"]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

# Define the GeminiAINode with API integration
import requests
from requests.exceptions import ConnectionError
def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

import base64
from langchain.llms import GooglePalm
from langchain.schema import HumanMessage

from langchain.llms import GooglePalm

class GeminiAINode:
    def run(self, object_data, depth_map, scene_data, image_path):
        # Prepare the data for the Gemini API
        data = {
            'objects': object_data.to_dict(),
            'depth_map': depth_map.tolist(),
            'scene_data': scene_data.tolist(),
            'image': image_path,  # You may need to convert the image for the API
        }
        json_data = json.dumps(data, indent=4)
        file_path = "data/sample_data.txt"
        with open(file_path, "w") as file:
            file.write(json_data)

        # Convert the image at image_path to base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Initialize the GooglePalm model via the LLM interface
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
        
        # Prepare the message with both text and the base64-encoded image
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe directions for how you would get from one side of the room to the other, and point out key obstacles. Specify the exact depth and distance between objects. Assume you were speaking to a blind person and helping them navigate."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"t"},
                },
            ],
        )

        # Invoke the Gemini AI model and get a response
        response = model([message])

        # Print the response content
        print(response.content)
        return response.content

class NavigationPipeline:
    def __init__(self):
        # Initialize each node
        self.object_detection_node = ObjectDetectionNode()
        self.depth_estimation_node = DepthEstimationNode()
        self.scene_analysis_node = SceneAnalysisNode()
        self.gemini_ai_node = GeminiAINode()

    def process(self, image_path):
        # Step 1: Object Detection
        objects = self.object_detection_node.run(image_path)
        
        # Step 2: Depth Estimation
        depth_map = self.depth_estimation_node.run(image_path)
        
        # Step 3: Scene Analysis
        scene_data = self.scene_analysis_node.run(image_path)
        
        # Step 4: Aggregation and Advice from Gemini AI
        advice, navigation_info = self.gemini_ai_node.run(objects, depth_map, scene_data, image_path)
        
        # Return final navigation advice
        return advice, navigation_info

# Example usage
def main(image_path):
    # Create the pipeline
    pipeline = NavigationPipeline()

    # Process the image and get navigation advice
    advice, navigation_info = pipeline.process(image_path)

    # Output the results
    print("Navigation Advice:", advice)
    print("Aggregated Information:", navigation_info)

# Example image path
image_path = 'test_image/maze.jpg'

main(image_path)
