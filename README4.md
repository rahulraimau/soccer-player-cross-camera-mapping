# Soccer Player Cross-Camera Mapping

This Python script performs cross-camera player mapping in soccer videos using YOLOv11 for player detection and ResNet50 for feature extraction. It matches players across broadcast and tacticam videos based on visual feature similarity.

## Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install opencv-python torch torchvision ultralytics numpy scikit-learn tqdm matplotlib pillow pandas
  ```
- A trained YOLOv11 model (`best.pt`) for player detection.
- Two input videos: `broadcast.mp4` and `tacticam.mp4`.

## Key Concepts

### Pillow
`Pillow` (PIL, Python Imaging Library) is a library for image processing. In this script, it converts OpenCV's BGR images to RGB format, as required by torchvision's transforms for ResNet input.

### tqdm
`tqdm` is a Python library that provides a progress bar for loops, showing the progress of tasks like video frame processing. It enhances user experience by visualizing the completion status of long-running operations.

### Bounding Box
A **bounding box** is a rectangular box defined by coordinates (x1, y1, x2, y2) that encloses an object (e.g., a player) in an image. In this script, YOLOv11 outputs bounding boxes around detected players, which are used to crop player regions for feature extraction.

**Conceptual Graph of a Bounding Box:**
```chartjs
{
  "type": "scatter",
  "data": {
    "datasets": [
      {
        "label": "Bounding Box",
        "data": [
          {"x": 100, "y": 100}, // Top-left (x1, y1)
          {"x": 200, "y": 100}, // Top-right
          {"x": 200, "y": 200}, // Bottom-right (x2, y2)
          {"x": 100, "y": 200}, // Bottom-left
          {"x": 100, "y": 100}  // Close the rectangle
        ],
        "borderColor": "#00FF00",
        "backgroundColor": "rgba(0, 255, 0, 0.2)",
        "showLine": true,
        "fill": true
      },
      {
        "label": "Player (Center)",
        "data": [{"x": 150, "y": 150}],
        "backgroundColor": "#FF0000",
        "pointRadius": 5
      }
    ]
  },
  "options": {
    "plugins": {
      "title": {
        "display": true,
        "text": "Bounding Box Around Player"
      }
    },
    "scales": {
      "x": {
        "min": 0,
        "max": 300,
        "title": { "display": true, "text": "X Coordinate" }
      },
      "y": {
        "min": 0,
        "max": 300,
        "title": { "display": true, "text": "Y Coordinate" },
        "reverse": true
      }
    }
  }
}
```
This chart shows a green bounding box (rectangle) around a red dot representing a player’s center in a 300x300 image. The coordinates (x1, y1) = (100, 100) and (x2, y2) = (200, 200) define the box.

### Cosine Similarity
**Cosine similarity** measures the similarity between two vectors by computing the cosine of the angle between them. It ranges from -1 (opposite) to 1 (identical). In this script, it compares feature vectors from ResNet50 to match players across videos. A higher score (e.g., above `SIMILARITY_THRESHOLD=0.7`) indicates a likely match.

**Formula**:  
\[ \text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} \]  
where \(\mathbf{A}\) and \(\mathbf{B}\) are feature vectors, \(\cdot\) is the dot product, and \(\|\mathbf{A}\|\) is the magnitude of \(\mathbf{A}\).

### PyTorch (torch) and Why `-1` is Used
**PyTorch** (`torch`) is a deep learning framework for building and running neural networks. In this script, it powers ResNet50 for feature extraction and manages tensor operations on CPU or GPU.

The `-1` in `list(feature_extractor.children())[:-1]` removes the final fully connected layer of ResNet50. ResNet50’s last layer outputs class probabilities (e.g., for ImageNet’s 1000 classes), but we need feature vectors (2048-dimensional) for similarity comparison. Removing the last layer extracts the feature map from the penultimate layer.

### Normalization Values (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
These values are used in the `transforms.Normalize` step to standardize RGB image channels for ResNet50. They are the mean and standard deviation of the ImageNet dataset’s RGB channels, on which ResNet50 was pretrained. Normalizing ensures the input matches the data distribution ResNet50 expects, improving feature extraction accuracy.

**Formula**:  
For each channel:  
\[ \text{Normalized} = \frac{\text{Input} - \text{Mean}}{\text{Std}} \]

## Code Overview
Below is a line-by-line explanation of the script, with line numbers and explanations of function parameters and logic.

### Imports
```python
1  import cv2
2  import torch
3  import numpy as np
4  from ultralytics import YOLO
5  from torchvision import models, transforms
6  from sklearn.metrics.pairwise import cosine_similarity
7  from tqdm import tqdm
8  import matplotlib.pyplot as plt
9  from PIL import Image
10 import pandas as pd
11 import os
```
- **Line 1-11**: Import libraries for image/video processing, deep learning, progress tracking, visualization, and data handling. See "Key Concepts" for `Pillow` and `tqdm`.

### Configuration
```python
12 YOLO_MODEL_PATH = "best.pt"
13 BROADCAST_VIDEO_PATH = r"C:\Users\DELL\Downloads\Soccer Player Cross-Camera Mapping\broadcast.mp4"
14 TACTICAM_VIDEO_PATH = r"C:\Users\DELL\Downloads\Soccer Player Cross-Camera Mapping\tacticam.mp4"
15 CONFIDENCE_THRESHOLD = 0.5
16 SIMILARITY_THRESHOLD = 0.7
17 DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
18 print(f"Using device: {DEVICE}")
```
- **Line 12-14**: Specify paths for the YOLO model and videos.
- **Line 15**: `CONFIDENCE_THRESHOLD` filters YOLO detections (detections with confidence < 0.5 are ignored).
- **Line 16**: `SIMILARITY_THRESHOLD` ensures only high-similarity matches (cosine similarity ≥ 0.7) are considered.
- **Line 17-18**: Selects GPU or CPU and prints the choice.

### YOLOv11 Model Loading
```python
19 try:
20     yolo_model = YOLO(YOLO_MODEL_PATH)
21     print(f"✅ YOLOv11 model '{YOLO_MODEL_PATH}' loaded successfully.")
22 except Exception as e:
23     print(f"❌ Error loading YOLOv11 model: {e}")
24     print("Please ensure 'best.pt' is in the correct directory and is a valid YOLOv11 model file.")
25     exit()
```
- **Line 19-25**: Loads the YOLOv11 model. Exits if the model file is invalid or missing.

### Feature Extractor Setup
```python
26 feature_extractor = models.resnet50(pretrained=True)
27 feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
28 feature_extractor.to(DEVICE)
29 feature_extractor.eval()
30 preprocess = transforms.Compose([
31     transforms.Resize((224, 224)),
32     transforms.ToTensor(),
33     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
34 ])
35 print("✅ Feature extractor (ResNet50) loaded and configured.")
```
- **Line 26-27**: Loads ResNet50 and removes the final layer for feature extraction (see "Why `-1` is Used").
- **Line 28-29**: Moves the model to the device and sets it to evaluation mode (no training).
- **Line 30-34**: Defines preprocessing: resizes images to 224x224, converts to tensors, and normalizes using ImageNet statistics.
- **Line 35**: Confirms setup.

### Feature Extraction from Image Crop
```python
36 def get_features_from_crop(image_crop):
37     if image_crop is None:
38         return None
39     image_crop_rgb = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
40     input_tensor = preprocess(image_crop_rgb)
41     input_batch = input_tensor.unsqueeze(0)
42     with torch.no_grad():
43         output = feature_extractor(input_batch.to(DEVICE))
44     return output.flatten().cpu().numpy()
```
- **Function Parameter**: `image_crop` (numpy array, BGR format from OpenCV).
- **Logic**: Converts the crop to RGB, preprocesses it, and extracts a feature vector using ResNet50.
- **Line 36-38**: Checks for a valid crop, returning None if invalid.
- **Line 39**: Uses `Pillow` to convert BGR to RGB.
- **Line 40-41**: Applies preprocessing and adds a batch dimension for ResNet.
- **Line 42-43**: Runs feature extraction without gradients for efficiency.
- **Line 44**: Flattens the output to a 1D vector and moves it to CPU.

### Video Feature Extraction
```python
45 def extract_features_from_video(video_path, model, confidence_threshold=0.5):
46     print(f"Starting feature extraction for: {video_path}")
47     cap = cv2.VideoCapture(video_path)
48     if not cap.isOpened():
49         print(f"❌ Error: Could not open video {video_path}")
50         return []
51     all_frame_data = []
52     frame_id = 0
53     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
54     pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")
55     while True:
56         ret, frame = cap.read()
57         if not ret:
58             break
59         frame_players = []
60         results = model(frame, verbose=False, conf=confidence_threshold)
61         if results and results[0].boxes:
62             for box in results[0].boxes:
63                 class_id = int(box.cls[0].item())
64                 if model.names[class_id] == 'player':
65                     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
66                     conf = float(box.conf[0].item())
67                     h, w, _ = frame.shape
68                     x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
69                     player_crop = frame[y1:y2, x1:x2]
70                     if player_crop.shape[0] > 0 and player_crop.shape[1] > 0:
71                         player_feature = get_features_from_crop(player_crop)
72                         if player_feature is not None:
73                             frame_players.append({
74                                 'box': [x1, y1, x2, y2],
75                                 'confidence': conf,
76                                 'feature': player_feature,
77                                 'original_frame_id': frame_id
78                             })
79         all_frame_data.append({'frame_id': frame_id, 'players': frame_players})
80         frame_id += 1
81         pbar.update(1)
82     cap.release()
83     pbar.close()
84     print(f"Finished feature extraction for: {video_path}. Processed {frame_id} frames.")
85     return all_frame_data
```
- **Function Parameters**:
  - `video_path`: Path to the video file.
  - `model`: YOLOv11 model for detection.
  - `confidence_threshold`: Minimum confidence for detections (default 0.5).
- **Logic**: Processes each video frame, detects players with YOLOv11, crops their bounding boxes, and extracts features.
- **Line 45-50**: Opens the video and checks for validity.
- **Line 51-54**: Initializes storage and a `tqdm` progress bar.
- **Line 55-58**: Reads frames until the video ends.
- **Line 59-60**: Runs YOLOv11 detection with the specified confidence threshold.
- **Line 61-64**: Filters detections for the 'player' class.
- **Line 65-68**: Extracts and bounds coordinates within the frame.
- **Line 69-70**: Crops the player region and checks for valid dimensions.
- **Line 71-72**: Extracts features and checks for validity.
- **Line 73-78**: Stores player data (bounding box, confidence, feature, frame ID).
- **Line 79-85**: Updates frame data, progress, and releases resources.

### Player Matching Across Videos
```python
86 def match_players(tacticam_frames_data, broadcast_frames_data, similarity_threshold=0.7, max_frame_diff=5):
87     print("Starting player matching...")
88     matches = []
89     global_player_id_counter = 0
90     tacticam_player_to_global_id = {}
91     broadcast_player_to_global_id = {}
92     final_matches_for_csv = []
93     for tacticam_frame_data in tqdm(tacticam_frames_data, desc="Matching frames"):
94         tacticam_frame_id = tacticam_frame_data['frame_id']
95         tacticam_players = tacticam_frame_data['players']
96         if not tacticam_players:
97             continue
98         candidate_broadcast_frames = []
99         for i in range(max(0, tacticam_frame_id - max_frame_diff), min(len(broadcast_frames_data), tacticam_frame_id + max_frame_diff + 1)):
100            candidate_broadcast_frames.append(broadcast_frames_data[i])
101        if not candidate_broadcast_frames:
102            continue
103        for i, tp in enumerate(tacticam_players):
104            best_match = None
105            max_similarity = -1
106            for broadcast_frame_data in candidate_broadcast_frames:
107                broadcast_frame_id = broadcast_frame_data['frame_id']
108                broadcast_players = broadcast_frame_data['players']
109                if not broadcast_players:
110                    continue
111                for j, bp in enumerate(broadcast_players):
112                    similarity = cosine_similarity(tp['feature'].reshape(1, -1), bp['feature'].reshape(1, -1))[0][0]
113                    if similarity > max_similarity and similarity >= similarity_threshold:
114                        max_similarity = similarity
115                        best_match = {
116                            'tacticam_frame': tacticam_frame_id,
117                            'tacticam_player_index': i,
118                            'broadcast_frame': broadcast_frame_id,
119                            'broadcast_player_index': j,
120                            'score': similarity
121                        }
122            if best_match:
123                tacticam_key = (best_match['tacticam_frame'], best_match['tacticam_player_index'])
124                broadcast_key = (best_match['broadcast_frame'], best_match['broadcast_player_index'])
125                if tacticam_key in tacticam_player_to_global_id:
126                    global_id = tacticam_player_to_global_id[tacticam_key]
127                elif broadcast_key in broadcast_player_to_global_id:
128                    global_id = broadcast_player_to_global_id[broadcast_key]
129                else:
130                    global_player_id_counter += 1
131                    global_id = global_player_id_counter
132                tacticam_player_to_global_id[tacticam_key] = global_id
133                broadcast_player_to_global_id[broadcast_key] = global_id
134                best_match['player_id'] = global_id
135                final_matches_for_csv.append(best_match)
136    print(f"Finished player matching. Found {len(final_matches_for_csv)} matches.")
137    return final_matches_for_csv
```
- **Function Parameters**:
  - `tacticam_frames_data`: List of tacticam frame data with player features.
  - `broadcast_frames_data`: List of broadcast frame data with player features.
  - `similarity_threshold`: Minimum cosine similarity for matches (default 0.7).
  - `max_frame_diff`: Temporal window for frame matching (default 5 frames).
- **Logic**: Matches players by comparing feature vectors within a temporal window, assigning consistent global IDs.
- **Line 86-92**: Initializes storage for matches and global IDs.
- **Line 93-97**: Iterates through tacticam frames, skipping empty ones.
- **Line 98-102**: Selects broadcast frames within `max_frame_diff`.
- **Line 103-105**: Iterates through tacticam players, tracking the best match.
- **Line 106-110**: Processes broadcast frames, skipping empty ones.
- **Line 111-112**: Computes cosine similarity (see "Cosine Similarity").
- **Line 113-121**: Updates the best match if similarity exceeds the threshold.
- **Line 122-135**: Assigns global IDs to maintain consistency across matches.
- **Line 136-137**: Prints and returns matches.

### Main Execution
```python
138 print("\n--- Testing YOLOv11 Detection ---")
139 if not os.path.exists(BROADCAST_VIDEO_PATH):
140     print(f"❌ Broadcast video not found at: {BROADCAST_VIDEO_PATH}")
141     exit()
142 cap_test = cv2.VideoCapture(BROADCAST_VIDEO_PATH)
143 ret_test, frame_test = cap_test.read()
144 cap_test.release()
145 if ret_test:
146     cv2.imwrite("sample_frame_from_broadcast.jpg", frame_test)
147     print("✅ Sample frame saved as 'sample_frame_from_broadcast.jpg'")
148     frame_display = cv2.imread("sample_frame_from_broadcast.jpg")
149     results_display = yolo_model(frame_display, verbose=False, conf=CONFIDENCE_THRESHOLD)
150     boxes_display = results_display[0].boxes.xyxy.cpu().numpy() if results_display[0].boxes else []
151     detected_players_count = 0
152     if results_display[0].boxes:
153         for box in results_display[0].boxes:
154             class_id = int(box.cls[0].item())
155             if yolo_model.names[class_id] == 'player':
156                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
157                 cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
158                 detected_players_count += 1
159     print(f"Detected {detected_players_count} players in sample frame (after confidence threshold).")
160     plt.figure(figsize=(10, 6))
161     plt.imshow(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
162     plt.title("YOLOv11 Player Detections on Sample Frame")
163     plt.axis("off")
164     plt.show()
165 else:
166     print("❌ Could not read frame from broadcast.mp4 for sample detection.")
```
- **Logic**: Tests YOLOv11 on a sample frame, visualizing bounding boxes.
- **Line 138-141**: Checks for the broadcast video and exits if missing.
- **Line 142-144**: Reads and releases a sample frame.
- **Line 145-147**: Saves the frame if read successfully.
- **Line 148-150**: Runs YOLOv11 detection on the saved frame.
- **Line 151-158**: Draws bounding boxes for 'player' detections and counts them.
- **Line 159-164**: Displays the frame with boxes using matplotlib.
- **Line 165-166**: Handles frame read failure.

```python
167 print("\n--- Extracting Features ---")
168 broadcast_feats = extract_features_from_video(BROADCAST_VIDEO_PATH, yolo_model, CONFIDENCE_THRESHOLD)
169 print(f"📦 Broadcast features extracted: {sum(len(f['players']) for f in broadcast_feats)} player detections across {len(broadcast_feats)} frames.")
170 tacticam_feats = extract_features_from_video(TACTICAM_VIDEO_PATH, yolo_model, CONFIDENCE_THRESHOLD)
171 print(f"📦 Tacticam features extracted: {sum(len(f['players']) for f in tacticam_feats)} player detections across {len(tacticam_feats)} frames.")
```
- **Line 167-171**: Extracts features from both videos and prints detection counts.

```python
172 print("\n--- Matching Players ---")
173 matches = match_players(tacticam_feats, broadcast_feats, SIMILARITY_THRESHOLD)
174 print("🔗 Matched players (unique matches found based on player_id logic):", len(matches))
```
- **Line 172-174**: Matches players and prints the number of matches.

```python
175 print("\n--- Saving Results ---")
176 if matches:
177     df = pd.DataFrame([{
178         "tacticam_frame": m["tacticam_frame"],
179         "broadcast_frame": m["broadcast_frame"],
180         "playerഗplayer_id": m["player_id"],
181         "similarity_score": m["score"]
182     } for m in matches])
183     df.to_csv("player_id_mapping.csv", index=False)
184     print("✅ Saved player_id_mapping.csv with", len(df), "entries.")
185 else:
186     print("⚠️ No matches found to save to CSV.")
187 print("\n--- Process Complete ---")
```
- **Line 175-187**: Saves matches to a CSV file with frame IDs, player IDs, and similarity scores, or prints a warning if no matches exist.

## Usage
1. Ensure `best.pt`, `broadcast.mp4`, and `tacticam.mp4` are in the specified paths.
2. Install dependencies listed in Prerequisites.
3. Run the script:
   ```bash
   python script.py
   ```
4. Check outputs:
   - `sample_frame_from_broadcast.jpg`: Sample frame with bounding boxes.
   - `player_id_mapping.csv`: Player matches with frame IDs and similarity scores.

## Notes
- The YOLO model must detect 'player' as a class.
- Adjust `CONFIDENCE_THRESHOLD` and `SIMILARITY_THRESHOLD` for performance.
- For robust tracking, consider DeepSORT for consistent player IDs before matching.