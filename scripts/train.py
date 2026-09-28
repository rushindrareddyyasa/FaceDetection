from ultralytics import YOLO


# ============================================================
# 1. LOAD A PRETRAINED YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# 2. TRAIN THE MODEL
# ============================================================

results = model.train(

    # Dataset configuration
    data="dataset.yaml",

    # Number of training iterations over dataset
    epochs=50,

    # Image size
    imgsz=640,

    # Number of images processed at once
    batch=4,

    # Name of experiment
    name="face_detection",

    # Folder where results are stored
    project="outputs"
)


print("Training completed!")