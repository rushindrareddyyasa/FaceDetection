from flask import Flask, render_template, request
import cv2
import numpy as np
import onnxruntime as ort
import os
import uuid


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# Load ONNX model

MODEL_PATH = "outputs/face_detection/weights/best.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# Letterbox image-

def letterbox(image, new_size=640):

    original_height, original_width = image.shape[:2]

    scale = min(
        new_size / original_width,
        new_size / original_height
    )

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

    canvas = np.full(
        (new_size, new_size, 3),
        114,
        dtype=np.uint8
    )

    pad_x = (new_size - new_width) // 2
    pad_y = (new_size - new_height) // 2

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    return canvas, scale, pad_x, pad_y


# IoU
def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


# Non-Maximum Suppression

def nms(boxes, scores, iou_threshold=0.45):

    indices = np.argsort(scores)[::-1]

    keep = []

    while len(indices) > 0:

        current = indices[0]

        keep.append(current)

        remaining = []

        for index in indices[1:]:

            iou = calculate_iou(
                boxes[current],
                boxes[index]
            )

            if iou < iou_threshold:
                remaining.append(index)

        indices = np.array(
            remaining,
            dtype=np.int64
        )

    return keep


# Detect faces

def detect_faces(image):

    original_height, original_width = image.shape[:2]

    # Prepare image
    input_image, scale, pad_x, pad_y = letterbox(
        image,
        640
    )

    # BGR → RGB
    input_image = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB
    )

    # uint8 → float32
    input_image = input_image.astype(
        np.float32
    ) / 255.0

    # HWC → CHW
    input_image = np.transpose(
        input_image,
        (2, 0, 1)
    )

    # Add batch dimension
    input_image = np.expand_dims(
        input_image,
        axis=0
    )

    # Run ONNX Runtime
    outputs = session.run(
        None,
        {
            input_name: input_image
        }
    )

    predictions = outputs[0]

    # Expected shape:
    # (1, 5, 8400)
    predictions = predictions[0].T

    boxes = []
    scores = []

    for prediction in predictions:

        center_x = prediction[0]
        center_y = prediction[1]

        width = prediction[2]
        height = prediction[3]

        confidence = prediction[4]

        if confidence < 0.25:
            continue

        # YOLO xywh → xyxy
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        # Remove letterbox padding
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clip coordinates
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))

        boxes.append([
            x1,
            y1,
            x2,
            y2
        ])

        scores.append(
            float(confidence)
        )

    if len(boxes) == 0:
        return []

    keep_indices = nms(
        boxes,
        scores,
        iou_threshold=0.45
    )

    detections = []

    for index in keep_indices:

        detections.append({
            "box": boxes[index],
            "confidence": scores[index]
        })

    return detections


# Flask route

@app.route("/", methods=["GET", "POST"])
def home():

    original_image = None
    cropped_images = []
    message = None

    if request.method == "POST":

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

        filename = str(uuid.uuid4()) + ".jpg"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(image_path)

        image = cv2.imread(image_path)

        if image is None:

            message = "Could not read the uploaded image."

            return render_template(
                "index.html",
                message=message
            )

        # Detect faces
        detections = detect_faces(image)

        face_count = 0

        for detection in detections:

            x1, y1, x2, y2 = detection["box"]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            face = image[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            face_count += 1

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

            cropped_images.append({
                "path": (
                    "/static/results/"
                    + result_filename
                ),
                "confidence": round(
                    detection["confidence"] * 100,
                    2
                ),
                "number": face_count
            })

        original_image = (
            "/static/uploads/"
            + filename
        )

        if face_count == 0:

            message = (
                "No face detected in the image."
            )

    return render_template(
        "index.html",
        original_image=original_image,
        cropped_images=cropped_images,
        message=message
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )