import cv2
from main_logic.detector import Detector
from main_logic.pose_estimator import posedetector

cap = cv2.VideoCapture('input/input_sample_video.mp4')
ret, frame = cap.read()

# Run detector first
detector = Detector()
detections = detector.detection_of_objects(frame)
print("Players found:", len(detections['players']))

# Then run pose estimator
pose_estimator = posedetector()

if detections['players']:
    player = detections['players'][0]
    landmarks = pose_estimator.get_landmarks(frame, player['box'])

    if landmarks:
        print("Pose detected!")
        print(f"Right wrist x: {landmarks[16].x:.2f}")
        print(f"Right wrist y: {landmarks[16].y:.2f}")
    else:
        print("No pose detected")

cap.release()
