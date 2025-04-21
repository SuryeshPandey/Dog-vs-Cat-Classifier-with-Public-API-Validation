import os
import torch
import cv2
import numpy as np
from config import IMAGE_DIR, PROCESSED_DIR
from PIL import Image, ImageEnhance
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.info(f"📂 Processed images will be saved in: {PROCESSED_DIR}")

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

def load_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        logging.info(f"✅ Loaded: {image_path}")
        return img
    except Exception as e:
        logging.error(f"❌ Error loading {image_path}: {e}")
        return None

def enhance_image(image, enhancement_type="brightness", factor=1.3):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = None
    if enhancement_type == "brightness":
        enhancer = ImageEnhance.Brightness(pil_image)
    elif enhancement_type == "contrast":
        enhancer = ImageEnhance.Contrast(pil_image)
    elif enhancement_type == "sharpness":
        enhancer = ImageEnhance.Sharpness(pil_image)

    if enhancer:
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    return image

def detect_objects(image):
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        results.render()
        detections = results.xywh[0].cpu().numpy()
        return detections, results.ims[0]
    except Exception as e:
        logging.error(f"❌ Detection error: {e}")
        return [], image

def process_image(image_path):
    img = load_image(image_path)
    if img is not None:
        img = enhance_image(img)
        detections, img_with_boxes = detect_objects(img)
        out_path = os.path.join(PROCESSED_DIR, "processed_" + os.path.basename(image_path))
        cv2.imwrite(out_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
        logging.info(f"💾 Saved: {out_path}")
        return out_path
    return None

def batch_process_images():
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    logging.info(f"📸 Found {len(images)} image(s) to process.")
    for fname in images:
        process_image(os.path.join(IMAGE_DIR, fname))

if __name__ == "__main__":
    batch_process_images()
