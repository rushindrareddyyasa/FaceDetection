import cv2
import os


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = "data_yolo/images/train"
LABEL_DIR = "data_yolo/labels/train"

OUTPUT_DIR = "outputs/visualized"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PROCESS IMAGES
# ============================================================

for filename in os.listdir(IMAGE_DIR):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    # Read image
    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(image_path)

    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Corresponding label file
    image_name = os.path.splitext(filename)[0]

    label_path = os.path.join(
        LABEL_DIR,
        image_name + ".txt"
    )

    # Read labels
    with open(label_path, "r") as file:

        lines = file.readlines()

    # Draw every bounding box
    for line in lines:

        values = line.strip().split()

        # YOLO format
        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # Convert normalized coordinates
        # back into pixel coordinates

        x_center = x_center * image_width
        y_center = y_center * image_height

        box_width = box_width * image_width
        box_height = box_height * image_height

        # Calculate corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)

        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Write class name
        cv2.putText(
            image,
            "face",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Save visualized image
    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        image
    )


print("Visualization completed!")
print("Check:", OUTPUT_DIR)