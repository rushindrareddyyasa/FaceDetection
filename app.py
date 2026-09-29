from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import uuid


app = Flask(__name__)

# Load trained YOLO model
model = YOLO("outputs/face_detection/weights/best.onnx")


# Folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():

    original_image = None
    cropped_images = []
    message = None

    if request.method == "POST":

        # Check whether image was uploaded
        if "image" not in request.files:
            message = "Please select an image."

            return render_template(
                "index.html",
                message=message
            )

        file = request.files["image"]

        if file.filename == "":
            message = "Please select an image."

            return render_template(
                "index.html",
                message=message
            )

        # Create unique filename
        filename = str(uuid.uuid4()) + ".jpg"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # Save uploaded image
        file.save(image_path)

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            message = "Could not read the uploaded image."

            return render_template(
                "index.html",
                message=message
            )

        # Resize very large images to reduce memory usage
        height, width = image.shape[:2]

        max_size = 640

        if max(height, width) > max_size:
            scale = max_size / max(height, width)

            new_width = int(width * scale)
            new_height = int(height * scale)

            image = cv2.resize(image, (new_width, new_height))

        # Run YOLO
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        face_count = 0

        # Process ALL detected faces
        for result in results:

            boxes = result.boxes

            for box in boxes:

                # Confidence score
                confidence = float(box.conf[0])

                # Ignore low-confidence detections
                if confidence < 0.5:
                    continue

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # Crop face
                face = image[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                # Increase face count
                face_count += 1

                # Save cropped face
                result_filename = (
                    f"face_{face_count}_{filename}"
                )

                result_path = os.path.join(
                    RESULT_FOLDER,
                    result_filename
                )

                cv2.imwrite(
                    result_path,
                    face
                )

                # Add result to list
                cropped_images.append({
                    "path": "/static/results/" + result_filename,
                    "confidence": round(confidence * 100, 2),
                    "number": face_count
                })

        original_image = "/static/uploads/" + filename

        if face_count == 0:

            message = "No face detected in the image."

    return render_template(
        "index.html",
        original_image=original_image,
        cropped_images=cropped_images,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)