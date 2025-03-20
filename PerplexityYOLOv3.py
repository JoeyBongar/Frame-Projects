import cv2
import numpy as np
import asyncio
from frame_sdk import Frame
from frame_sdk.camera import Quality, AutofocusType
from frame_sdk.display import Alignment, PaletteColors
from PIL import Image
import io
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from scipy.spatial.distance import cosine
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from coco_label_map import LABEL_MAP
import urllib.request
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Load YOLOv3 model and configuration
model_cfg = "/Users/I572464/Library/CloudStorage/OneDrive-SAPSE/Desktop/frameTest/.venv/yolov3.cfg"
model_weights = "/Users/I572464/Library/CloudStorage/OneDrive-SAPSE/Desktop/frameTest/.venv/yolov3.weights"
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

# Load COCO dataset class labels
with open("/Users/I572464/Library/CloudStorage/OneDrive-SAPSE/Desktop/frameTest/.venv/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the input image dimensions for YOLO
input_width, input_height = 416, 416

async def capture_image_from_glasses(frame):
    """
    Capture an image from the Frame AR glasses camera.
    The image is returned in JPEG format as bytes.
    """
    # Configure camera settings
    # async frame.camera.take_photo(autofocus_seconds: Optional[int] = 3, quality: Quality = Quality.MEDIUM, autofocus_type: AutofocusType = AutofocusType.AVERAGE) -> bytes:

    frame.auto_process_photo = False  # Disable auto-processing (optional)
    autofocus_seconds = 3  # Autofocus duration (set to None to disable)
    quality =  Quality.HIGH # Set photo quality (LOW, MEDIUM, HIGH, FULL)
    autofocus_type = AutofocusType.AVERAGE  # Autofocus type (AVERAGE, SPOT, CENTER_WEIGHTED)
    
    # Take a photo with the specified settings
    photo_data = await frame.camera.take_photo(
        autofocus_seconds=autofocus_seconds,
        quality=quality,
        autofocus_type=autofocus_type
    )
    
    # Decode the JPEG bytes into an OpenCV image
    np_arr = np.frombuffer(photo_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    return image

def detect_objects(image):
    """
    Detect objects in the provided image using YOLOv3.
    """
    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Perform forward pass
    detections = net.forward(output_layers)
    
    height, width = image.shape[:2]
    boxes, confidences, class_ids = [], [], []
    
    # Process detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            results.append({"label": label, "confidence": confidence, "box": (x, y, w, h)})
            
            # Draw bounding box and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
    
    return results

async def main():
    """
    Main function to capture an image from AR glasses and detect objects.
    """
    async with Frame() as frame:
        print("Connected to Frame. Starting camera feed...")
        while True:
            try:
                # Capture an image from AR glasses
                photo_bytes = await capture_image_from_glasses(frame)
                
                # Detect objects in the captured image
                detections = detect_objects(photo_bytes)
                
                # Display results
                print("Detected Objects:")
                for obj in detections:
                    print(f"Label: {obj['label']}, Confidence: {obj['confidence']:.2f}")
                highest_confidence_obj = max(detections, key=lambda obj: obj['confidence'])
                await frame.display.show_text(highest_confidence_obj['label'], align=Alignment.MIDDLE_CENTER)
                # frame.display.show_text(highest_confidence_obj['label'])
                
                # Show the processed image with bounding boxes
                cv2.imshow("Detected Objects", photo_bytes)
                cv2.waitKey(0)
                cv2.destroyAllWindows()    
                   
            except Exception as e:
                print(f"An error occurred: {e}")
            await asyncio.sleep(0.5)     
    print("Disconnected from Frame.")
if __name__ == "__main__":
    asyncio.run(main())

