from ultralytics import YOLO
from PIL import Image

# Load YOLOv5 model (auto-downloads if not present)
model = YOLO("yolov5s.pt")

def detect_objects(image_path):
    img = Image.open(image_path)
    results = model(img)      # List of results
    result = results[0]       # Get the first result
    df = result.to_df()       # Convert detections to DataFrame
    return df, result
