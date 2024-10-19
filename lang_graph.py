import langgraph as lg

import torch

class ObjectDetectionNode:
    def run(self, image):
        # Load YOLOv5 model for object detection
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        results = model(image)
        objects = results.pandas().xyxy[0]  # bounding boxes and labels
        return objects[['name', 'xmin', 'ymin', 'xmax', 'ymax']]
# Define the depth estimation node
import torch
from PIL import Image
import numpy as np

class DepthEstimationNode:
    def run(self, image_path):
        # Load MiDaS model for depth estimation
        model_type = "DPT_Large"  # or "MiDaS_small" for a smaller model
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        image = Image.open(image_path)
        input_image = midas_transforms(image).unsqueeze(0)
        with torch.no_grad():
            prediction = midas(input_image)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        return depth_map
import torch
import clip
from PIL import Image

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
class GeminiAINode:
    def run(self, object_data, depth_map, scene_data):
        # Aggregation logic
        important_objects = object_data[object_data['name'].isin(['person', 'car', 'bicycle'])]
        
        navigation_info = {
            'objects': important_objects,
            'depth_map': depth_map.mean(),  # Overall depth
            'scene': scene_data  # Scene probabilities
        }
        
        if 'person' in important_objects['name'].values:
            advice = "Caution: Person ahead. Keep right."
        elif 'car' in important_objects['name'].values:
            advice = "Caution: Car ahead. Move to the side."
        else:
            advice = "Path is clear. Proceed straight."
        
        return advice, navigation_info
# Define the full LangGraph pipeline
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
        advice, navigation_info = self.gemini_ai_node.run(objects, depth_map, scene_data)
        
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
image_path = 'test_image/maze.png'

main(image_path)
