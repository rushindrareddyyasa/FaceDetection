import os
import shutil
import random
import xml.etree.ElementTree as ET


# PATHS

# Original training images
TRAIN_IMAGE_DIR = "data/images/train"

# Original training XML annotations
TRAIN_ANNOTATION_DIR = "data/annotations/train"

# Original test images
TEST_IMAGE_DIR = "data/images/test"

# Original test XML annotations
TEST_ANNOTATION_DIR = "data/annotations/test"

# YOLO dataset directories
YOLO_TRAIN_IMAGE_DIR = "data_yolo/images/train"
YOLO_VAL_IMAGE_DIR = "data_yolo/images/val"
YOLO_TEST_IMAGE_DIR = "data_yolo/images/test"

YOLO_TRAIN_LABEL_DIR = "data_yolo/labels/train"
YOLO_VAL_LABEL_DIR = "data_yolo/labels/val"
YOLO_TEST_LABEL_DIR = "data_yolo/labels/test"


# CREATE DIRECTORIES

directories = [
    YOLO_TRAIN_IMAGE_DIR,
    YOLO_VAL_IMAGE_DIR,
    YOLO_TEST_IMAGE_DIR,
    YOLO_TRAIN_LABEL_DIR,
    YOLO_VAL_LABEL_DIR,
    YOLO_TEST_LABEL_DIR
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)


# XML → YOLO CONVERSION FUNCTION

def convert_xml_to_yolo(xml_path):
    """
    Read XML file
    and convert all bounding boxes to YOLO format.
    """

    # Read XML file
    tree = ET.parse(xml_path)

    # Get root element
    root = tree.getroot()

    # Get image width and height
    size = root.find("size")

    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    yolo_annotations = []

    # Find every object in the XML
    for object_element in root.findall("object"):

        # Get bounding box
        bbox = object_element.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Calculate bounding-box width and height
        box_width = xmax - xmin
        box_height = ymax - ymin

        # Calculate center coordinates
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        # Normalize coordinates
        x_center = x_center / image_width
        y_center = y_center / image_height
        box_width = box_width / image_width
        box_height = box_height / image_height

        # Class 0 = face
        class_id = 0

        # YOLO format:
        # class_id x_center y_center width height
        annotation = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        yolo_annotations.append(annotation)

    return yolo_annotations


# GET TRAINING IMAGES
all_images = []

for filename in os.listdir(TRAIN_IMAGE_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_name = os.path.splitext(filename)[0]

        xml_path = os.path.join(
            TRAIN_ANNOTATION_DIR,
            image_name + ".xml"
        )

        # Only use images that have annotations
        if os.path.exists(xml_path):

            all_images.append(filename)

        else:

            print(
                f"Skipping {filename}: "
                f"annotation file not found."
            )


print()
print("Total annotated training images:", len(all_images))


# SHUFFLE DATA

# Fixed seed means we get the same split every time.
random.seed(42)

random.shuffle(all_images)


# TRAIN / VALIDATION SPLIT

split_index = int(len(all_images) * 0.8)

train_images = all_images[:split_index]
val_images = all_images[split_index:]


print("Training images:", len(train_images))
print("Validation images:", len(val_images))


# COPY TRAINING / VALIDATION DATA
def prepare_split(image_list, image_output_dir, label_output_dir):

    for image_filename in image_list:

        # Original image path
        source_image = os.path.join(
            TRAIN_IMAGE_DIR,
            image_filename
        )

        # Destination image path
        destination_image = os.path.join(
            image_output_dir,
            image_filename
        )

        # Copy image
        shutil.copy2(
            source_image,
            destination_image
        )

        # XML path
        image_name = os.path.splitext(image_filename)[0]

        xml_path = os.path.join(
            TRAIN_ANNOTATION_DIR,
            image_name + ".xml"
        )

        # Convert XML → YOLO
        yolo_annotations = convert_xml_to_yolo(xml_path)

        # Label file
        label_path = os.path.join(
            label_output_dir,
            image_name + ".txt"
        )

        # Save YOLO labels
        with open(label_path, "w") as file:

            for annotation in yolo_annotations:
                file.write(annotation + "\n")


# Prepare training data
prepare_split(
    train_images,
    YOLO_TRAIN_IMAGE_DIR,
    YOLO_TRAIN_LABEL_DIR
)


# Prepare validation data
prepare_split(
    val_images,
    YOLO_VAL_IMAGE_DIR,
    YOLO_VAL_LABEL_DIR
)


# PREPARE TEST DATA

test_images = []

for filename in os.listdir(TEST_IMAGE_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_name = os.path.splitext(filename)[0]

        xml_path = os.path.join(
            TEST_ANNOTATION_DIR,
            image_name + ".xml"
        )

        if os.path.exists(xml_path):

            test_images.append(filename)

        else:

            print(
                f"Skipping test image {filename}: "
                f"annotation not found."
            )


print("Test images:", len(test_images))


for image_filename in test_images:

    # Copy test image
    source_image = os.path.join(
        TEST_IMAGE_DIR,
        image_filename
    )

    destination_image = os.path.join(
        YOLO_TEST_IMAGE_DIR,
        image_filename
    )

    shutil.copy2(
        source_image,
        destination_image
    )

    # XML
    image_name = os.path.splitext(image_filename)[0]

    xml_path = os.path.join(
        TEST_ANNOTATION_DIR,
        image_name + ".xml"
    )

    # Convert annotation
    yolo_annotations = convert_xml_to_yolo(xml_path)

    # Save label
    label_path = os.path.join(
        YOLO_TEST_LABEL_DIR,
        image_name + ".txt"
    )

    with open(label_path, "w") as file:

        for annotation in yolo_annotations:
            file.write(annotation + "\n")


print()
print("Dataset preparation completed!")