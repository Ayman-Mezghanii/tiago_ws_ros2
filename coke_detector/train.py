from ultralytics import YOLO

# Load a model
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # You can start with the 'n' version for quick training

# Train the model
model.train(data='coke.yaml', epochs=50, imgsz=640)
model.export(format='onnx')

