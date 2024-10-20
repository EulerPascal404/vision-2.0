import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import clip
from dotenv import load_dotenv
import os
import requests
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Your Gemini API key from .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Function to encode an image in base64
def encode_image(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
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
            x_center = (obj['xmin'] + obj['xmax']) / 2
            y_center = (obj['ymin'] + obj['ymax']) / 2
            depth_value = depth_map[int(y_center), int(x_center)]
            object_depth_info.append({
                "name": obj['name'],
                "depth": round(depth_value, 2),
                "location": f"centered at ({int(x_center)}, {int(y_center)})",
                "x_center": x_center,
                "y_center": y_center,
                "xmin": obj['xmin'],
                "ymin": obj['ymin'],
                "xmax": obj['xmax'],
                "ymax": obj['ymax']
            })
        
        return object_depth_info

    def summarize_scene(self, object_depth_info):
        summary = []
        for obj in object_depth_info:
            summary.append(f"{obj['name']} at depth {obj['depth']} (location: {obj['location']})")
        return "; ".join(summary)

    def run(self, object_data, depth_map, scene_data, image_path, navigation_mode):
        image = Image.open(image_path)
        object_depth_info = self.combine_data(object_data, depth_map, image.size)
        scene_summary = self.summarize_scene(object_depth_info)

        if navigation_mode:
            prompt_text = "describe directions for how you would get from one side of the room to the other, and point out key obstacles. Specify the exact depth and distance between objects. Assume you were speaking to a blind person and helping them navigate. Provide navigation advice based on the following:"
        else:
            prompt_text = "Provide a general summary of the scene and the objects to watch out for. Specify their locations and general distances from each other. Assume you are speaking to a blind person, and describe what is in the scene to watch out for."

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": f"Scene Summary: {scene_summary}"},
                {"type": "text", "text": "Scene Analysis probabilities: " + str(scene_data.tolist())},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
            ]
        )
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
                return response.content, object_depth_info
            else:
                print("Invalid response or content not available.")
        except Exception as e:
            print(f"Error invoking the Gemini AI model: {e}")
            return None, None

# Navigation Pipeline
class NavigationPipeline:
    def __init__(self):
        self.object_detection_node = ObjectDetectionNode()
        self.depth_estimation_node = DepthEstimationNode()
        self.scene_analysis_node = SceneAnalysisNode()
        self.gemini_ai_node = GeminiAINode()

    def process(self, image_path, navigation_mode):
        objects = self.object_detection_node.run(image_path)
        depth_map = self.depth_estimation_node.run(image_path)
        scene_data = self.scene_analysis_node.run(image_path)
        advice, object_depth_info = self.gemini_ai_node.run(objects, depth_map, scene_data, image_path, navigation_mode)
        return advice, object_depth_info

# Physics-based function to calculate the path with obstacle avoidance
def trace_navigation_path(advice, draw, image_size, object_depth_info):
    """Trace the complete path based on the directions in advice, incorporating physics dynamics and bounding boxes."""
    # Start at the bottom-center of the image
    current_position = np.array([image_size[0] // 2, image_size[1] - 50])
    velocity = np.array([0.0, 0.0])  # Initialize velocity as float

    # Extract bounding box data (obstacle avoidance)
    obstacles = [(obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']) for obj in object_depth_info]

    # Define direction vectors
    direction_vectors = {
        "left": np.array([-150, 0]),
        "right": np.array([150, 0]),
        "straight": np.array([0, -200]),
        "forward": np.array([0, -200]),
    }
    
    # Physics parameters
    acceleration = 10  # Arbitrary acceleration value
    max_velocity = 40  # Max velocity limit
    repulsion_strength = 1000  # Strength of repulsive force from obstacles
    safe_distance = 50  # Minimum safe distance from obstacles (pixels)

    def calculate_repulsion_force(current_pos):
        """Calculate the repulsion force from nearby obstacles."""
        total_force = np.array([0, 0])
        for (xmin, ymin, xmax, ymax) in obstacles:
            center_obstacle = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
            distance_vector = current_pos - center_obstacle
            distance = np.linalg.norm(distance_vector)
            
            if distance < safe_distance:
                # Repulsive force is inversely proportional to the distance (closer objects = stronger repulsion)
                force_magnitude = repulsion_strength / (distance**2 + 1e-5)  # Add small value to prevent division by 0
                force_direction = distance_vector / distance  # Normalize the direction
                total_force += force_magnitude * force_direction
                
        return total_force

    # Parse advice and trace the path
    directions = advice.lower().split(",")
    for direction in directions:
        direction = direction.strip()

        # Apply the force-based dynamic system for each direction
        for key, vector in direction_vectors.items():
            if key in direction:
                # Calculate the next target position
                target_position = current_position + vector

                # Physics: adjust velocity with acceleration and cap it at max_velocity
                velocity += vector / np.linalg.norm(vector) * acceleration
                velocity = np.clip(velocity, -max_velocity, max_velocity)

                # Avoid obstacles: calculate repulsive force from nearby obstacles
                repulsion_force = calculate_repulsion_force(current_position)
                velocity += repulsion_force

                # Update the position with the new velocity
                next_position = current_position + velocity

                # Draw the 3D arrow from the current position to the next position
                draw_3d_arrow(draw, tuple(current_position), tuple(next_position))
                current_position = next_position

def draw_3d_arrow(draw, start_pos, end_pos, color="red", width=10):
    """Draw a 3D-like arrow from start_pos to end_pos."""
    # Draw the arrow's body (thicker to give a 3D effect)
    draw.line([start_pos, end_pos], fill=color, width=width)
    
    # Draw the arrowhead at the end position
    arrow_size = 15
    draw.line([end_pos, (end_pos[0] - arrow_size, end_pos[1] + arrow_size)], fill=color, width=width)
    draw.line([end_pos, (end_pos[0] + arrow_size, end_pos[1] + arrow_size)], fill=color, width=width)

    # Add a 3D shading effect
    shading_color = "darkred"
    shadow_offset = 5
    draw.line([start_pos, (end_pos[0] + shadow_offset, end_pos[1] + shadow_offset)], fill=shading_color, width=width // 2)

# Function to annotate the image and generate voice directions
def annotate_image_with_directions_and_voice(image_path, object_depth_info, advice):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Draw bounding boxes for objects
    for obj in object_depth_info:
        draw.rectangle(
            [(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax'])], outline="green", width=3
        )
        draw.text((obj['xmin'], obj['ymin'] - 10), obj['name'], fill="green", font=font)

    # Trace the complete path from the navigation advice
    trace_navigation_path(advice, draw, image.size, object_depth_info)

    # Save the annotated image
    annotated_image_path = "test_image/annotated_" + os.path.basename(image_path)
    image.save(annotated_image_path)

    # Call Vapi.ai Text-to-Speech API to generate voice directions
    voice_response = generate_voice_directions(advice)
    return annotated_image_path, voice_response

# Function to generate voice directions using Vapi.ai
def generate_voice_directions(advice):
    api_key = os.getenv('VAPI_API_KEY')  # Replace with your Vapi.ai API key
    url = "https://api.vapi.ai/v1/text-to-speech"
    payload = {
        "text": advice,
        "voice": "en-US-Wavenet-D",  # Example voice
        "speed": 1.0,
        "pitch": 1.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        audio_file = "navigation_instructions.mp3"
        with open(audio_file, "wb") as f:
            f.write(response.content)
        return audio_file
    else:
        print("Error generating voice directions:", response.text)
        return None

# Example usage
def main(image_path, string_num, navigation_mode=True):
    # Create directory 'nav_data' if it doesn't exist
    output_dir = "nav_data_" + string_num
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pipeline = NavigationPipeline()
    advice, object_depth_info = pipeline.process(image_path, navigation_mode)

    # Output the navigation advice
    print("Navigation Advice:", advice)

    # Annotate the image and generate voice directions
    annotated_image, voice_response = annotate_image_with_directions_and_voice(image_path, object_depth_info, advice)

    # Save the annotated image in the 'nav_data' directory
    annotated_image_filename = os.path.join(output_dir, "annotated_image.jpg")
    os.rename(annotated_image, annotated_image_filename)
    print("Annotated image saved to:", annotated_image_filename)

    # Save the navigation advice to a text file in the 'nav_data' directory
    advice_filename = os.path.join(output_dir, "navigation_advice.txt")
    with open(advice_filename, "w") as advice_file:
        advice_file.write(advice)
    print("Navigation advice saved to:", advice_filename)

    # Save the voice response (MP3 file) in the 'nav_data' directory
    if voice_response:
        voice_response_filename = os.path.join(output_dir, "navigation_instructions.mp3")
        os.rename(voice_response, voice_response_filename)
        print("Voice directions saved to:", voice_response_filename)
    else:
        print("No voice directions generated.")


# Example image path
image_path = 'demo_images/demo_image_5.jpeg'
main(image_path, "5", navigation_mode=True)
