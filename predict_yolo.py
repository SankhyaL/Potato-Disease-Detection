from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLOv8 model once
yolo_model = YOLO("models/best.pt")

def predict(image_file):
    """Run YOLOv8 model for detection/segmentation"""
    # Open the image
    image = Image.open(image_file)
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, "JPEG")
        tmp_path = tmp.name

    results = yolo_model(tmp_path)
    res = results[0]

    # Extract labels from detections
    if len(res.boxes) > 0:
        labels = [yolo_model.names[int(cls)] for cls in res.boxes.cls]
        return f"Detected: {', '.join(set(labels))}"
    else:
        return "No disease detected"
