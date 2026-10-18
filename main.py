import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run inference on an image
results = model("https://ultralytics.com/images/bus.jpg")

# Save result image
for r in results:
    r.save(filename="output.jpg")

print("Detection completed and saved to output.jpg")
