# soccer-player-cross-camera-mapping

⚽ Soccer Player Re-Identification (Cross-Camera Matching)

Match and track soccer players across two different video views (broadcast & tacticam) using deep learning-based detection and appearance similarity.

🔍 Project Summary

This project performs cross-camera player matching in soccer footage by:

📦 Detecting players in each frame using YOLOv11

🧠 Extracting visual features from player crops using ResNet50

📊 Matching players across camera views using cosine similarity

🧾 Assigning player IDs and exporting results to CSV

🎯 Use Case: Useful for sports analytics, tactical performance analysis, or highlight automation.

🧰 Tech Stack

Tool

Purpose

YOLOv11

Player detection (object detection)

ResNet50

Feature extraction (appearance descriptor)

Cosine Similarity

Matching players by feature direction

OpenCV

Video processing

PyTorch

Model loading & GPU support

tqdm, pandas

Progress & data handling

🚀 Workflow

Load YOLOv11 weights (best.pt) and detect players frame-by-frame.

Crop each player, convert to RGB, and feed into ResNet50 to extract 2048-dim vector.

Match each tacticam player to broadcast view using cosine similarity:

If similarity > 0.7 → players are matched

Assigns global player_id to both views

Export CSV of matched frame pairs with ID and similarity score.

📁 File Structure

.
├── best.pt                         # YOLOv11 trained model
├── broadcast.mp4                   # Broadcast camera video
├── tacticam.mp4                    # Tacticam camera video
├── main.py                         # Main script
├── sample_frame_from_broadcast.jpg# Sample detection image
├── player_id_mapping.csv          # Final output matches
└── README.md

📊 Output Format (CSV)

tacticam_frame,broadcast_frame,player_id,similarity_score
12,13,1,0.8123
15,14,2,0.7761
...

📸 Sample Frame Preview

Runs YOLOv11 on a sample frame

Draws bounding boxes for player verification

Saved as sample_frame_from_broadcast.jpg

⚙️ Setup & Run

Install Dependencies

pip install ultralytics torch torchvision opencv-python scikit-learn pandas tqdm matplotlib

Run the Code

python main.py

📈 Future Improvements

Add DeepSORT or ByteTrack for temporal tracking

Recognize jersey numbers for fine-grained ID

Use 3D keypoints or pose descriptors

Real-time camera calibration + sync

📄 License

This project is licensed under the MIT License. See LICENSE file for full terms.
