# 🛍️ YOLOv8 Mall Crowd Detection

This project uses the YOLOv8 object detection model to analyze mall surveillance footage and detect people for crowd estimation. It leverages the Mall Dataset with YOLO-formatted labels and provides video-based results showing detected individuals in real time.

---

## 📂 Project Structure
yolov8-mall-detection/ ├── detect_video.py # Script to run YOLOv8 detection on video ├── mall_dataset/ # Original mall dataset images ├── mall_yolo_labels/ # YOLOv8 formatted label files ├── requirements.txt # Python dependencies └── README.md # Project description
## 🚀 How to Run
1. **Clone the repository**  
   ```bash
   git clone https://github.com/siva053491/yolov8-mall-detection.git
   cd yolov8-mall-detection
Install dependencies:
python detect_video.py
output:
from IPython.display import HTML
from base64 import b64encode

mp4 = open('output_detected.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML(f"""
<video width=640 controls>
    <source src="{data_url}" type="video/mp4">
</video>
""")
📁 Dataset
Mall Dataset: Original dataset of mall surveillance frames.
YOLO Labels: Converted annotations in YOLO format.
✨ Features
Real-time crowd detection using YOLOv8
Video-based visualization
Easy to run and extend
Siva053491
GitHub: github.com/siva053491


