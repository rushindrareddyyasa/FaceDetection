from ultralytics import YOLO

# Load our trained face detection model
model = YOLO("outputs/face_detection/weights/best.pt")

# Export to ONNX
model.export(
    format="onnx",
    imgsz=640,
    simplify=False
)

print("ONNX export completed!")