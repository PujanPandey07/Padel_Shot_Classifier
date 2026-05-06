class ShotClassifier:
    def __init__(self):
        self.prev_ball_x = None
        self.shot_cooldown = 0

    def is_shot_moment(self, ball_detection):
        """Detect when a shot occurs based on ball movement"""
        if ball_detection is None:
            return False

        bx1, by1, bx2, by2 = ball_detection['box']
        ball_x = (bx1 + bx2) / 2

        shot_detected = False
        if self.prev_ball_x is not None and self.shot_cooldown == 0:
            if abs(ball_x - self.prev_ball_x) > 20:
                shot_detected = True
                self.shot_cooldown = 15

        self.prev_ball_x = ball_x
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1

        return shot_detected

    def classify(self, landmarks, frame_shape):
        """Classify shot type based on pose landmarks"""
        if landmarks is None:
            return "unknown"

        h, w = frame_shape[:2]

        # Get key landmarks
        r_wrist = landmarks[16]
        r_shoulder = landmarks[12]
        l_shoulder = landmarks[11]
        r_elbow = landmarks[14]

        # Calculate body center
        body_center_x = (r_shoulder.x + l_shoulder.x) / 2

        # Pixel coordinates for vertical comparison
        wrist_y = r_wrist.y * h
        shoulder_y = r_shoulder.y * h

        # YOUR classification logic
        if wrist_y < shoulder_y - 0.05 * h:
            return "smash"
        elif r_wrist.x > body_center_x:
            return "forehand"
        else:
            return "backhand"
