import os
import shutil
import random
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm

# Local data directory
data_dir = "PlantVillage"

# Project directory for YOLO
project_dir = "yolo_project"
images_train_dir = os.path.join(project_dir, "images", "train")
images_val_dir = os.path.join(project_dir, "images", "val")
labels_train_dir = os.path.join(project_dir, "labels", "train")
labels_val_dir = os.path.join(project_dir, "labels", "val")

for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
    os.makedirs(d, exist_ok=True)

# Get classes
classes = sorted(os.listdir(data_dir))
class_map = {cls: idx for idx, cls in enumerate(classes)}
print("Classes found:", classes)

# Collect all images
all_images = []
for cls in classes:
    cls_dir = os.path.join(data_dir, cls)
    if os.path.isdir(cls_dir):
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(cls_dir, img_name))

random.seed(42)
random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

def make_yolo_label(img_path, label_path, class_id):
    """Create YOLO label with bounding box"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("cv2.imread returned None")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        white_ratio = (th == 255).sum() / (w * h)
        if white_ratio < 0.01:
            th = cv2.bitwise_not(th)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            x, y, ww, hh = 0, 0, w, h
        else:
            c = max(contours, key=cv2.contourArea)
            x, y, ww, hh = cv2.boundingRect(c)
            if (ww * hh) < 0.01 * (w * h):
                x, y, ww, hh = 0, 0, w, h

        x_center = (x + ww / 2) / w
        y_center = (y + hh / 2) / h
        w_norm = ww / w
        h_norm = hh / h

        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    except Exception as e:
        print(f"Warning: auto-label failed for {img_path} ({e}). Using full-image box.")
        x_center, y_center, w_norm, h_norm = 0.5, 0.5, 1.0, 1.0
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def copy_and_label(img_list, images_dest_dir, labels_dest_dir):
    for img_path in tqdm(img_list):
        parent = os.path.basename(os.path.dirname(img_path))
        class_id = class_map[parent]
        fname = os.path.basename(img_path)
        dest_img_path = os.path.join(images_dest_dir, fname)
        dest_label_path = os.path.join(labels_dest_dir, os.path.splitext(fname)[0] + ".txt")
        shutil.copy2(img_path, dest_img_path)
        make_yolo_label(dest_img_path, dest_label_path, class_id)

print("Auto-labeling and copying train images...")
copy_and_label(train_imgs, images_train_dir, labels_train_dir)

print("Auto-labeling and copying val images...")
copy_and_label(val_imgs, images_val_dir, labels_val_dir)

# Create data.yaml
data_yaml_path = os.path.join(project_dir, "data.yaml")
names_yaml_list = "[" + ", ".join([f"'{c}'" for c in classes]) + "]"

# Use absolute paths
abs_images_train_dir = os.path.abspath(images_train_dir)
abs_images_val_dir = os.path.abspath(images_val_dir)

data_yaml = f"""train: {abs_images_train_dir}
val: {abs_images_val_dir}
nc: {len(classes)}
names: {names_yaml_list}
"""

with open(data_yaml_path, "w") as f:
    f.write(data_yaml)

print("Wrote data.yaml")

# Train YOLO model
model = YOLO('yolov8n.pt')
result = model.train(
    data=data_yaml_path,
    epochs=30,
    imgsz=320,
    batch=16,
    device='cpu',
    name='potato_yolov8',
    project=project_dir
)

# Save the best model
import glob
best_paths = glob.glob(os.path.join(project_dir, "**", "best.pt"), recursive=True)
if best_paths:
    best_weight = best_paths[-1]
    shutil.move(best_weight, "models/best.pt")
    print("Model saved as models/best.pt")
else:
    print("No best.pt found")
