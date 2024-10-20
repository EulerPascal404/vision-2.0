import torch
import torch.nn.functional as F  # Import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import clip
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Your Gemini API key from .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def encode_image(image_path: str) -> str:
    try:
        # Open the image from the file path
        image = Image.open(image_path)

        # Convert the image to a byte stream (JPEG format)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

        # Encode the image in base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding the image: {e}")
        return ""

# Object Detection Node
class ObjectDetectionNode:
    def run(self, image):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        results = model(image)
        objects = results.pandas().xyxy[0]
        return objects[['name', 'xmin', 'ymin', 'xmax', 'ymax']]

# Depth Estimation Node
class DepthEstimationNode:
    def run(self, image_path):
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0)
        with torch.no_grad():
            prediction = midas(input_image)
            prediction = F.interpolate(
                prediction.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        return depth_map

# Scene Analysis Node
class SceneAnalysisNode:
    def run(self, image_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(["a street", "a park", "a building", "a house", "trees"]).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

# Gemini AI Node
class GeminiAINode:
    def combine_data(self, object_data, depth_map, image_shape):
        object_depth_info = []
        height, width = image_shape
        
        for _, obj in object_data.iterrows():
            # Get the object's bounding box coordinates
            x_center = (obj['xmin'] + obj['xmax']) / 2
            y_center = (obj['ymin'] + obj['ymax']) / 2
            
            # Estimate the depth at the object's center point
            depth_value = depth_map[int(y_center), int(x_center)]
            
            # Store object name and depth info
            object_depth_info.append({
                "name": obj['name'],
                "depth": round(depth_value, 2),
                "location": f"centered at ({int(x_center)}, {int(y_center)})"
            })
        
        return object_depth_info

    def summarize_scene(self, object_depth_info):
        summary = []
        for obj in object_depth_info:
            summary.append(f"{obj['name']} at depth {obj['depth']} (location: {obj['location']})")
        return "; ".join(summary)

    def run(self, object_data, depth_map, scene_data, image_path):
        # Combine object segmentation with depth analysis
        image = Image.open(image_path)
        object_depth_info = self.combine_data(object_data, depth_map, image.size)

        # Summarize scene for Gemini API
        scene_summary = self.summarize_scene(object_depth_info)
        
        # Concise prompt message
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe directions for how you would get from one side of the room to the other, and point out key obstacles. Specify the exact depth and distance between objects. Assume you were speaking to a blind person and helping them navigate. Provide navigation advice based on the following:"},
                {"type": "text", "text": f"Scene Summary: {scene_summary}"},
                {"type": "text", "text": "Scene Analysis probabilities: " + str(scene_data.tolist())},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
            ]
        )

        # Initialize Gemini AI Model and invoke it
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.5,
            max_tokens=500,
            max_retries=2,
        )
        
        try:
            response = llm.invoke([message])
            if response and hasattr(response, 'content'):
                return response.content
            else:
                print("Invalid response or content not available.")
        except Exception as e:
            print(f"Error invoking the Gemini AI model: {e}")
            return None

# Navigation Pipeline
class NavigationPipeline:
    def __init__(self):
        self.object_detection_node = ObjectDetectionNode()
        self.depth_estimation_node = DepthEstimationNode()
        self.scene_analysis_node = SceneAnalysisNode()
        self.gemini_ai_node = GeminiAINode()

    def process(self, image_path):
        objects = self.object_detection_node.run(image_path)
        depth_map = self.depth_estimation_node.run(image_path)
        scene_data = self.scene_analysis_node.run(image_path)
        advice = self.gemini_ai_node.run(objects, depth_map, scene_data, image_path)
        return advice

# Example usage
def main(image_path):
    pipeline = NavigationPipeline()
    advice = pipeline.process(image_path)
    print("Navigation Advice:", advice)

# Example image path
image_path = 'test_image/hotel_room.jpg'
main(image_path)
