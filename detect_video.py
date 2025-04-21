

import zipfile

with zipfile.ZipFile("mall_dataset.zip", "r") as zip_ref:
    zip_ref.extractall("mall_data")

print("✅ mall_dataset.zip extracted!")

import os, shutil

src = "mall_data/mall_dataset/frames"
dst = "renamed_frames"
os.makedirs(dst, exist_ok=True)

for fname in os.listdir(src):
    if fname.startswith("seq_") and fname.endswith(".jpg"):
        num = int(fname.split("_")[1].split(".")[0])
        new_name = f"frame_{num:04d}.jpg"
        shutil.copy(os.path.join(src, fname), os.path.join(dst, new_name))

print("✅ All images renamed to frame_XXXX.jpg")

import os
print("Sample label files:", os.listdir("mall_yolo_labels")[:5])


import zipfile

with zipfile.ZipFile("mall_yolo_labels.zip", 'r') as zip_ref:
    zip_ref.extractall("mall_yolo_labels")

print("✅ Labels extracted to mall_yolo_labels/")


import os
print("Sample label files:", os.listdir("mall_yolo_labels")[:5])


image_dir = "renamed_frames"
label_dir = "mall_yolo_labels"  # now this folder should exist!

images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
labels = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

# Match names
image_names = set([f.replace(".jpg", "") for f in images])
label_names = set([f.replace(".txt", "") for f in labels])
both = image_names & label_names

print(f"✅ Matched image-label pairs: {len(both)}")


from sklearn.model_selection import train_test_split
import shutil
import os

# Paths
image_dir = "renamed_frames"
label_dir = "mall_yolo_labels"
output_base = "mall_yolo_dataset"

# Get matching pairs
images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
labels = [f.replace(".jpg", ".txt") for f in images if os.path.exists(os.path.join(label_dir, f.replace(".jpg", ".txt")))]
valid_images = [f.replace(".txt", ".jpg") for f in labels]

# Split: 70% train, 20% val, 10% test
train, temp = train_test_split(valid_images, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=1/3, random_state=42)

# Create folders and copy files
for split, data in {"train": train, "val": val, "test": test}.items():
    os.makedirs(f"{output_base}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_base}/labels/{split}", exist_ok=True)
    for fname in data:
        shutil.copy(os.path.join(image_dir, fname), f"{output_base}/images/{split}/{fname}")
        shutil.copy(os.path.join(label_dir, fname.replace(".jpg", ".txt")), f"{output_base}/labels/{split}/{fname.replace('.jpg', '.txt')}")

print("✅ Dataset split completed: train, val, test.")

from pathlib import Path

dataset_path = Path("mall_yolo_dataset").resolve()

data_yaml = f"""
path: {dataset_path}
train: images/train
val: images/val

names:
  0: person
"""

with open("data.yaml", "w") as f:
    f.write(data_yaml)

print("✅ data.yaml created at:", dataset_path / "data.yaml")

import pandas as pd

results_path = "runs/detect/mall_crowd_detector/results.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    print(df.tail())  # Show last few rows (latest epochs)
else:
    print("Training results not found.")




from ultralytics import YOLO

# Load your best model (trained on crowd data)
model = YOLO("runs/detect/mall_crowd_detector/weights/best.pt")


# Replace 'crowd.mp4' with your uploaded video name
model.predict(
    source="sam.mp4",      # or "test_image.jpg" if using image
    conf=0.3,                # confidence threshold
    save=True,               # saves output video/image with boxes
    save_txt=False,          # save labels (True if needed)
    device='cpu'             # ensure CPU mode
)





from ultralytics import YOLO

# Load your best model (trained on crowd data)
model = YOLO("runs/detect/mall_crowd_detector/weights/best.pt")

# Replace 'crowd.mp4' with your uploaded video name
model.predict(
    source="Screenshot 2025-04-18 201622.png",      # or "test_image.jpg" if using image
    conf=0.3,                # confidence threshold
    save=True,               # saves output video/image with boxes
    save_txt=False,          # save labels (True if needed)
    device='cpu'             # ensure CPU mode
)



from ultralytics import YOLO

# Load your best model (trained on crowd data)
model = YOLO("runs/detect/mall_crowd_detector/weights/best.pt")

# Replace 'crowd.mp4' with your uploaded video name
model.predict(
    source="sample.mp4",      # or "test_image.jpg" if using image
    conf=0.3,                # confidence threshold
    save=True,               # saves output video/image with boxes
    save_txt=False,          # save labels (True if needed)
    device='cpu'             # ensure CPU mode
)


from ultralytics import YOLO

# Load your best model (trained on crowd data)
model = YOLO("runs/detect/mall_crowd_detector/weights/best.pt")

# Replace 'crowd.mp4' with your uploaded video name
model.predict(
    source="3002466-hd_1920_1080_25fps.mp4",      # or "test_image.jpg" if using image
    conf=0.3,                # confidence threshold
    save=True,               # saves output video/image with boxes
    save_txt=False,          # save labels (True if needed)
    device='cpu'             # ensure CPU mode
)