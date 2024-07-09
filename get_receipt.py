from ultralytics import YOLO
import cv2
import math
import os

def detect_and_crop(image_path, target_class_name, output_dir="cropped"):
    # Load the YOLO v8 model
    model = YOLO("yolo-Weights/yolov8n.pt")

    # Object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    # Read the image using OpenCV
    img = cv2.imread(image_path)
    results = model(img)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("confidence: ", confidence)
            # Class name
            cls = int(box.cls[0])
            print("class name:", classNames[cls])
            if cls < len(classNames) and classNames[cls] == target_class_name and confidence > 0.2:
                # Draw a rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # cropped image
                cropped_img = img[y1:y2, x1:x2]
                # save the cropped image
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, f"{target_class_name}_cropped.jpg")
                cv2.imwrite(output_path, cropped_img)
                print(f"Cropped image saved at: {output_path}")
                # return the path of the saved image
                return output_path
    print("No target class detected or confidence too low.")
    return None

# # Example usage
# image_path = 'images/image2.jpg'
# target_class_name = "book"  # YOLO v8 identifies any receipt as book
# cropped_image_path = detect_and_crop(image_path, target_class_name)