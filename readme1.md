# ⚽ Soccer Player Cross-Camera Mapping using YOLOv11 + ResNet50

> Match and identify the same soccer players captured from two different camera angles (broadcast & tacticam) using object detection and deep feature similarity.

---

## 📽️ Project Overview

In this project, we:
- Detect players in two video feeds using **YOLOv11**
- Extract visual features using **ResNet50**
- Match the same players across both cameras based on **cosine similarity**
- Save the player ID mappings into a CSV file for analysis

🎯 **Goal**: Enable automated cross-view player tracking in sports videos.

---

## 🛠️ Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| YOLOv11 | Player detection |
| ResNet50 (PyTorch) | Feature extraction |
| OpenCV | Video & image handling |
| Scikit-learn | Cosine similarity |
| Pandas | Data processing |
| tqdm | Progress bars |

---

## 🧱 Project Structure

```bash
.
├── broadcast.mp4                   # Broadcast camera video
├── tacticam.mp4                    # Tacticam video
├── best.pt                         # YOLOv11 trained weights
├── sample_frame_from_broadcast.jpg# Frame preview
├── player_id_mapping.csv          # Final matched players
├── main.py                         # Main execution script
└── README.md
🔄 Workflow
Load YOLOv11 to detect players in each frame

Extract deep features (ResNet50) from each player's crop

Compare features across videos using cosine similarity

Assign global player IDs to matched pairs

Export results to player_id_mapping.csv

📊 Output Sample
csv
Copy
Edit
tacticam_frame,broadcast_frame,player_id,similarity_score
12,13,1,0.8123
15,14,2,0.7761
...
🧪 How to Run
🔧 Place your video files and best.pt YOLO model in the same directory.

📦 Install dependencies:

bash
Copy
Edit
pip install ultralytics torch torchvision opencv-python scikit-learn tqdm pandas matplotlib
▶️ Run the main script:

bash
Copy
Edit
python main.py
📸 Sample Detection Frame


🧠 Future Improvements
Add player tracking (e.g., DeepSORT)

Recognize jersey numbers

Sync frames more accurately

Real-time live match integration

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

yaml
Copy
Edit
