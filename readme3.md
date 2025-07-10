# ⚽ Soccer Player Cross-Camera Mapping using YOLOv11 + ResNet50

> Match soccer players across two different video angles (broadcast and tacticam) using object detection, deep feature extraction, and cosine similarity.

---

## 📽️ Project Overview

This project helps in automatically **identifying the same soccer player** captured from two different camera views:

- 📺 **Broadcast camera** (TV view)
- 🎥 **Tacticam camera** (Top-down or tactical view)

We use:

- 🧠 **YOLOv11** for detecting players
- 🧠 **ResNet50** to extract visual features of players
- 🧮 **Cosine Similarity** to match same players across views

---

## 📦 Code Walkthrough

### 🔹 1. **Imports & Configuration**

```python
import cv2, torch, numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
Purpose: Brings in all necessary libraries for video processing, model loading, vector comparison, and visualization.

🔹 2. Model & Device Setup
python
Copy
Edit
YOLO_MODEL_PATH = "best.pt"
BROADCAST_VIDEO_PATH = r"broadcast.mp4"
TACTICAM_VIDEO_PATH = r"tacticam.mp4"
CONFIDENCE_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.7
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Sets the path for YOLO weights & input videos

Defines thresholds for detection and similarity

Chooses between GPU (CUDA) or CPU for performance

🔹 3. Load YOLOv11 Model
python
Copy
Edit
yolo_model = YOLO(YOLO_MODEL_PATH)
Loads the custom YOLOv11 weights (best.pt) to detect players in frames.

🔹 4. Feature Extractor (ResNet50)
python
Copy
Edit
feature_extractor = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.to(DEVICE)
Loads a pretrained ResNet50 model

Removes the final classification layer

Used to generate 2048-dimensional feature vectors for player crops

🔹 5. Image Preprocessing Function
python
Copy
Edit
def get_features_from_crop(image_crop):
    ...
Converts YOLO crop (OpenCV BGR) to PIL Image (RGB)

Applies normalization & resizing

Extracts a feature vector via ResNet

🔹 6. Video Feature Extraction
python
Copy
Edit
def extract_features_from_video(video_path, model, confidence_threshold=0.5):
    ...
Opens the video

For each frame:

Detects players using YOLO

Crops player regions

Extracts deep features

Stores data frame-by-frame with features, coordinates, and confidence

🔹 7. Player Matching via Cosine Similarity
python
Copy
Edit
def match_players(tacticam_frames_data, broadcast_frames_data, similarity_threshold=0.7):
    ...
Compares each player in the tacticam frame with players from nearby broadcast frames

Uses cosine_similarity() to compare 2048-dim feature vectors

Matches are accepted if score > 0.7

Global IDs are assigned to link the same player across both views

🔹 8. Detection Preview (Optional)
python
Copy
Edit
cap_test = cv2.VideoCapture(BROADCAST_VIDEO_PATH)
ret_test, frame_test = cap_test.read()
...
Saves and displays a frame from the broadcast video

Runs YOLO on it and shows bounding boxes

Helps verify if the YOLO model is working correctly

🔹 9. Main Execution Flow
python
Copy
Edit
broadcast_feats = extract_features_from_video(...)
tacticam_feats = extract_features_from_video(...)
matches = match_players(...)
Runs all core functions to:

Extract detections and features

Match players

Save final matches

🔹 10. Save Matches to CSV
python
Copy
Edit
df.to_csv("player_id_mapping.csv", index=False)
Final result shows which players matched across views, including:

tacticam_frame

broadcast_frame

player_id

similarity_score

📊 Sample Output
csv
Copy
Edit
tacticam_frame,broadcast_frame,player_id,similarity_score
12,14,1,0.8023
16,17,2,0.7654
...
🧪 Dependencies
bash
Copy
Edit
pip install ultralytics torch torchvision opencv-python scikit-learn pandas tqdm matplotlib
📁 Directory Structure
bash
Copy
Edit
.
├── best.pt                         # YOLO model weights
├── broadcast.mp4                   # Broadcast video
├── tacticam.mp4                    # Tacticam video
├── main.py                         # Full script
├── sample_frame_from_broadcast.jpg# Sample preview frame
├── player_id_mapping.csv          # Output match results
├── README.md
📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

