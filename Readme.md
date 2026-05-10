# Padel Game Analytics — Shot Classification System

### Layman AI Internship Assignment

**Author:** Pujan Pandey  
**Date:** May 2026

---

## What This Project Does

This system takes a padel match video and automatically detects when players hit shots, then classifies each shot as a Forehand, Backhand, or Smash. It outputs an annotated video with labels overlaid on players, along with structured JSON and CSV files containing all detected shots with timestamps.

---

## Project Structure

```
padel-shot-classifier/
│
├── src/
│   ├── detector.py          # YOLOv8 player and ball detection
│   ├── pose_estimator.py    # YOLOv8-pose keypoint extraction
│   ├── ball_tracker.py      # Ball in play tracking logic
│   ├── shot_classifier.py   # Shot detection and classification
│   └── utils.py             # Helper functions + optical flow
│
├──/ input/
│   └── input_sample_video.mp4
|   |___data/
        |
│       ├── 2022_BCN_FinalF_1_ball.json
│       ├── 2022_BCN_FinalF_1_pose.json
│       ├── 2022_BCN_FinalF_1_shots.csv
│       ├── 2022_BCN_FinalM_1_ball.json
│       ├── 2022_BCN_FinalM_1_pose.json
│       └── 2022_BCN_FinalM_1_shots.csv
│
|
├── output/
│   ├── output_video.mp4
│   ├── shot_results.json
│   └── shot_results.csv
│
├── models/
│   ├── yolov8s.pt
│   ├── yolov8s-pose.pt
│   ├── shot_classifier.pkl
│   └── shot_scaler.pkl
│
├── train_classifier.py      # Training script
├── main.py                  # Main inference pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone repository
git clone https://github.com/PujanPandey07/Padel_Shot_Classifier.git
cd Padel_Shot_Classifier

# 2. Create virtual environment
python -m venv env
env\Scripts\activate  # Windows
source env/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place input video
# Put your video at input/input_sample_video.mp4

# 5. Place dataset labels (for training)
# data/labels/2022_BCN_FinalF_1_shots.csv
# data/labels/2022_BCN_FinalF_1_pose.json
# data/labels/2022_BCN_FinalM_1_shots.csv
# data/labels/2022_BCN_FinalM_1_pose.json

# 6. Train classifier
python train_classifier.py

# 7. Run inference
python main.py
```

---

## Requirements

---

## Output Format

### JSON

```json
[
  {
    "frame": 245,
    "timestamp": "00:08.16",
    "player_id": 1,
    "shot_type": "Forehand"
  },
  {
    "frame": 412,
    "timestamp": "00:13.73",
    "player_id": 2,
    "shot_type": "Backhand"
  }
]
```

### CSV

---

## Demo and Resources

- **Demo Video:** [Google Drive Link]
- **Model Weights:** [Google Drive Link]
- **Sample Output JSON:** [Google Drive Link]

---

## My Approach — The Full Story

I want to be honest about this submission. This was a genuinely challenging problem and I went through many approaches, each one teaching me something new about the real difficulties of sports analytics. Here is the complete journey.

---

### The Core Challenge

The input video is recorded from a **top down CCTV style camera** overlooking the court. This created fundamental challenges throughout the project because most computer vision models and datasets for sports analytics assume a **side angle broadcast camera**. Almost every problem I encountered came back to this root cause.

---

### Stage 1 — Basic Pipeline Setup

I started by building the foundational detection pipeline using:

- **YOLOv8** (Ultralytics) for player detection
- **OpenCV** for frame by frame video processing
- **ByteTrack** for player tracking across frames

Player detection worked reasonably well — 2 to 4 players detected per frame depending on occlusion. Ball detection was immediately problematic. The padel ball is very small from a top down angle and YOLOs standard COCO weights (class 32 — sports ball) rarely detected it reliably.

---

### Stage 2 — Ball Detection Problem

I tried three approaches for ball detection:

**Approach A — YOLO class 32**
Standard pretrained weights occasionally detected the ball but with very low confidence and many false positives. Multiple balls are visible on a padel court — balls in active play plus spare balls sitting on the ground — making it hard to identify which one is the active ball.

**Approach B — Color based detection**
Padel balls are light green/yellow. I used HSV color space filtering to detect circular green objects. This worked partially but produced many false positives — court markings, player clothing, and lighting reflections all triggered detections.

**Approach C — Movement based filtering**
Combined color detection with movement tracking across frames. A ball in play moves between frames while stationary balls on the ground do not. This improved results but was still unreliable because the color detection itself was noisy.

**What I learned:** Ball detection from a top down fixed camera is genuinely difficult. A paper I found during research — _Padel Analytics using Deep Learning_ (Arutjothi et al., 2025) — specifically used TrackNet, a deep learning model designed for tracking small fast moving sports objects, to solve this problem. TrackNet uses heatmap prediction across consecutive frames rather than single frame bounding box detection, making it much more robust to motion blur and small object size. This is the correct solution I would implement with more time.

---

### Stage 3 — Player Tracking Problem

**DeepSORT attempt:**
I initially implemented DeepSORT for player tracking — the same algorithm I used in my final year person re-identification project at university. DeepSORT uses a Kalman filter for motion prediction combined with appearance features for re-identification across frames.

However padel players change direction rapidly and unpredictably. The Kalman filter assumes linear motion and predicted positions drifted badly away from actual player positions during direction changes. Player IDs were jumping from 1, 2 to 54, 57 — completely unstable.

**Switched to ByteTrack:**
ByteTrack uses IoU based matching without motion prediction, associating detections across frames based on bounding box overlap rather than predicted trajectories. This is significantly better for sports with rapid direction changes. IDs became much more stable after this switch.

**Position based slot system:**
Even with ByteTrack, ID switching still occurred. I implemented a position based slot system — instead of trusting YOLO track IDs, the system assigns stable slots (0, 1, 2, 3) based on court position. A player consistently in the top left area of court gets slot 0 regardless of what ID ByteTrack assigns. This gave stable per-player state tracking for shot detection.

---

### Stage 4 — Shot Detection

Without reliable ball detection, I needed another signal to detect when a shot happens. I tried several approaches in sequence:

**Approach A — Ball direction change**
When the ball changes direction a shot has occurred. Theoretically correct but required reliable ball tracking which I did not have.

**Approach B — Single threshold wrist speed**
If the right wrist moves faster than a threshold between consecutive processed frames, a shot is happening. This produced too many false positives — normal running and walking also move the wrist fast.

**Approach C — Multi condition detection**
Required three conditions simultaneously:

- Wrist speed above threshold
- Wrist moving faster than elbow (arm swing, not body movement)
- Wrist moving faster than hip (not just general running)

This reduced false positives but MediaPipe landmark instability on the top down camera made wrist coordinates noisy.

**Approach D — Optical flow combined with wrist speed**
Added optical flow as an additional signal. A shot creates distinctive concentrated fast motion in the player region. I compute mean optical flow magnitude in the player bounding box using cv2.calcOpticalFlowFarneback and combine it with wrist speed. Only when both signals are elevated simultaneously does the system register a shot. This was the most reliable approach and is the current implementation.

**Approach E — Trained binary shot detector**
Used the PadelTracker100 dataset frame labels to train a binary classifier — is this a shot frame or not. This gave the most meaningful detections by learning from 45000 labeled real padel frames.

---

### Stage 5 — Shot Classification

Classifying forehand vs backhand vs smash was the most difficult part of this project.

**Rule based approach:**
Started with simple geometric rules:

- Wrist significantly above shoulder → smash
- Wrist right of body center → forehand
- Wrist left of body center → backhand

This was inaccurate because from a top down camera the vertical dimension is compressed, shoulder and wrist heights appear similar, and left/right positioning is harder to interpret. MediaPipe was also not designed for overhead views.

**Switched from MediaPipe to YOLOv8-pose:**
I discovered that the PadelTracker100 dataset was annotated using YOLOv8-pose for keypoint extraction. By switching our inference pipeline from MediaPipe to YOLOv8-pose we aligned the feature extraction method between training data and inference — the same model family extracting keypoints in both cases. This was an important realisation about the source of classification errors.

**ML based approach with PadelTracker100 dataset:**
I found the PadelTracker100 dataset (Novillo et al., 2024) which contains 45,934 labeled frames from a female match and 53,953 from a male match, with frame by frame annotations of shot type, ball position, and player pose keypoints.

I trained a GradientBoosting classifier on 18 biomechanical features extracted from pose keypoints. All features are normalized by shoulder width to be scale invariant:

| Feature Group  | Features                                         |
| -------------- | ------------------------------------------------ |
| Wrist position | Right and left wrist x,y relative to body center |
| Elbow position | Right and left elbow x,y relative to body center |
| Wrist height   | Right and left wrist height above shoulder       |
| Arm extension  | Right and left arm extension ratio               |
| Body rotation  | Shoulder angle                                   |
| Torso          | Torso height ratio                               |
| Wrist to hip   | Right wrist x,y relative to hip                  |
| Elbow angles   | Right and left elbow bend angles                 |

Training accuracy achieved 60% on held out test data across Forehand, Backhand, and Smash classes.

**The domain gap problem:**
Inference accuracy was lower than training accuracy for a specific reason I identified. The PadelTracker100 dataset was recorded with a **side angle broadcast camera** while our test video uses a **top down surveillance camera**. The same physical shot produces completely different pose keypoint values from these two camera angles. Wrist height relative to shoulder looks very different from directly above compared to from the side. This is called domain shift and is a known challenge in deploying sports analytics models across different camera setups. I addressed this partially by normalizing all features by shoulder width, but the fundamental camera angle difference remains the limiting factor.

---

### Stage 6 — Using PadelTracker100 Ball Annotations

The dataset includes precise ball position annotations per frame. I used these to improve shot detection — when the annotated ball position is close to a detected player, it provides a strong signal that a shot is occurring. This combines ground truth ball data with our pose based wrist speed detection for more reliable shot triggering.

---

## What Works

- Player detection on main court using YOLOv8
- Stable player slot assignment using position based tracking
- Ball in play isolation from multiple court balls
- Shot event detection using optical flow and wrist velocity combined
- Shot classification using GradientBoosting trained on PadelTracker100
- Structured JSON and CSV output with timestamps
- Annotated output video with color coded labels — green for forehand, orange for backhand, red for smash

---

## Known Limitations and Why

**Ball detection accuracy:**
The padel ball is very small from a top down angle and moves fast causing motion blur. Standard YOLO misses it frequently. Color detection picks up false positives. The correct solution is TrackNet trained on padel footage.

**Far side player classification:**
Players on the far side of the court appear smaller in frame. YOLOv8-pose struggles to extract accurate keypoints for small detections, leading to missed classifications or no shot detections on that half of the court.

**Classification accuracy:**
60% on training data distribution, lower in practice due to the camera angle domain gap described above. The fundamental issue is that no publicly available labeled dataset uses the same top down surveillance camera angle as our test video.

**Shot detection false positives:**
Fast running and jumping can occasionally trigger shot detection even without ball contact. The optical flow threshold helps filter these but does not eliminate them completely.

---

## What I Would Improve With More Time

**1. TrackNet for ball detection**
Specifically designed for small fast sports objects. Uses heatmap prediction across 3 consecutive frames. Would solve the ball detection problem properly.

**2. Custom labeled dataset**
Label 500 to 1000 frames from the actual top down video for training. This would eliminate the domain gap entirely — training and inference from the same camera angle.

**3. Temporal sequence model**
An LSTM or Transformer processing 20 frame sequences of keypoints rather than single frames. This captures the full motion arc of a shot rather than a snapshot, which is more physically meaningful. A shot is a motion event not a single pose.

**4. Action recognition model**
Models like SlowFast or VideoMAE process video clips directly end to end. No separate detection and classification steps — the model learns what a shot looks like from the video itself.

**5. Court homography**
Apply perspective transformation to normalize the court view, correcting for the camera angle. This would make positions and angles consistent regardless of where players are on the court.

**6. Racket detection**
Fine tune a YOLO model to detect padel rackets specifically. Racket position relative to player body is a very reliable signal for shot type — forehand has racket on the dominant side, backhand has it crossing the body.

---

## Literature Referenced

**Arutjothi G., Thrishaa R., Vidhya S. (2025)**
_Padel Analytics using Deep Learning_
IJIRMPS Volume 13 Issue 2
Used YOLO for player detection, TrackNet for ball tracking, ResNet-50 for gesture classification. This paper validated our technology choices and clearly identified TrackNet as the correct approach for ball tracking in padel. It also confirmed that pose based gesture classification is the right direction for shot type detection.

**Novillo Á., Aceña V., Lancho C., Cuesta M., De Diego I.M. (2024)**
_Padel Two-Dimensional Tracking Extraction from Monocular Video Recordings_
IDEAL 2024, Lecture Notes in Computer Science, vol 15346
Source of the PadelTracker100 dataset used for training our classifier. 99,887 labeled frames across two professional matches with shot type, ball position, and player pose annotations.

**Huang Y.C. et al. (2019)**
_TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications_
IEEE AVSS 2019
Motivated understanding of why standard object detection fails for small fast sports balls and the heatmap based approach needed to solve it properly.

**Redmon J., Farhadi A. (2018)**
_YOLOv3: An Incremental Improvement_
University of Washington
Foundation of the YOLO family used throughout this project for detection and pose estimation.

---

## Tech Stack

| Component           | Technology                           |
| ------------------- | ------------------------------------ |
| Player detection    | YOLOv8s (Ultralytics)                |
| Pose estimation     | YOLOv8s-pose                         |
| Player tracking     | ByteTrack                            |
| Ball tracking       | Color detection + movement filtering |
| Shot detection      | Optical flow + wrist velocity        |
| Shot classification | GradientBoosting (scikit-learn)      |
| Training data       | PadelTracker100 dataset              |
| Video processing    | OpenCV                               |
| Data processing     | NumPy, Pandas                        |

---

## Honest Reflection

This assignment was genuinely difficult and I want to be transparent about that. The top down camera angle created a domain gap that affected every component of the system. I tried many approaches — rule based geometry, wrist velocity thresholds, DeepSORT tracking, MediaPipe pose, trained ML classifiers — and each one taught me something real about the challenges of sports analytics in practice.

The most valuable learning was understanding WHY things failed. MediaPipe is not designed for overhead views. YOLO struggles with small fast moving objects. Domain shift between broadcast camera training data and surveillance camera inference data fundamentally limits accuracy. Kalman filters assume linear motion which padel players violate constantly.

I believe the path I took — trying multiple methods, identifying root causes of failures, finding and using real labeled data, making informed decisions about tradeoffs — reflects how real AI engineering work happens. The system is not perfect but it is honest, documented, and built with genuine understanding of the problem.
