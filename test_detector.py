# test_detector.py (create this in root folder temporarily)
import cv2
from main_logic.detector import Detector

cap = cv2.VideoCapture('input/input_sample_video.mp4')
ret, frame = cap.read()

detector = Detector()
detections = detector.detection_of_objects(frame)

print("Players found:", len(detections['players']))
print("Ball found:", detections['ball'])

cap.release()
