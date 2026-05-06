import mediapipe as mp
import cv2


class posedetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def get_landmarks(self, frame, player_box):
        """Extract pose landmarks from a cropped player region"""
        x1, y1, x2, y2 = [int(c) for c in player_box]

        # Crop player region from frame
        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return None

        # MediaPipe needs RGB, OpenCV gives BGR
        rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_crop)

        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def draw_landmarks(self, frame, landmarks, player_box):
        """Draw skeleton on frame for visualization"""
        x1, y1, x2, y2 = [int(c) for c in player_box]
        player_crop = frame[y1:y2, x1:x2]

        self.mp_drawing.draw_landmarks(
            player_crop,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        frame[y1:y2, x1:x2] = player_crop
