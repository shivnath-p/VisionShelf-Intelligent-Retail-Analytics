from ultralytics import YOLO
from PIL import Image

# Load YOLOv5 model (auto-downloads yolov5s.pt if not present)
model = YOLO("yolov5s.pt")

def detect_objects(image_path):
    img = Image.open(image_path)
    results = model(img)             # run inference
    result = results[0]              # get first result
    df = result.pandas().xyxy[0]     # dataframe of detections
    return df, result
