import cv2
import os
from ultralytics import YOLO


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = YOLO(
    "outputs/face_detection/weights/best.pt"
)


# ============================================================
# 2. INPUT IMAGE
# ============================================================

image_path = "test.jpg"


# ============================================================
# 3. READ IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:

    print("Could not read image.")

    exit()


# ============================================================
# 4. RUN FACE DETECTION
# ============================================================

results = model(image)


# ============================================================
# 5. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs/cropped_faces",
    exist_ok=True
)


face_count = 0


# ============================================================
# 6. PROCESS DETECTED FACES
# ============================================================

for result in results:

    # Get detected boxes
    boxes = result.boxes

    for box in boxes:

        # Confidence score
        confidence = float(
            box.conf[0]
        )

        # Ignore weak detections
        if confidence < 0.5:
            continue

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]

        # Convert to integers
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        print(
            f"Face detected: "
            f"confidence={confidence:.2f}"
        )

        print(
            f"Bounding box: "
            f"({x1}, {y1}, {x2}, {y2})"
        )


        # ====================================================
        # CROP FACE
        # ====================================================

        face = image[
            y1:y2,
            x1:x2
        ]


        # ====================================================
        # SAVE FACE
        # ====================================================

        face_count += 1

        output_path = (
            f"outputs/cropped_faces/"
            f"face_{face_count}.jpg"
        )

        cv2.imwrite(
            output_path,
            face
        )

        print(
            "Saved:",
            output_path
        )


print()
print(
    f"Total faces detected: {face_count}"
)