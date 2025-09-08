from ultralytics import YOLO

def detect_objects(image_path):
    model = YOLO("yolov8n.pt")  # or your custom-trained model
    print("Loaded classes:", model.names)  # 👈 show what classes the model knows

    results = model(image_path)

    for result in results:
        df = result.to_df()

        # Keep all objects above confidence 0.5
        df = df[df['confidence'] > 0.5]

        print("Detected objects:", len(df))
        return df, result
