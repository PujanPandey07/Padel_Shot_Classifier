class ShotClassifier:
    def __init__(self):
        self.ball_history = []
        self.shot_cooldown = 0

    def is_shot_moment(self, ball_detection, frame_shape, players=None):
        """Detect when a shot occurs based on ball speed and proximity"""
        if ball_detection is None:
            return False

        h, w = frame_shape[:2]
        bx1, by1, bx2, by2 = ball_detection['box']
        ball_x = (bx1 + bx2) / 2
        ball_y = (by1 + by2) / 2

        # Maintain a short history to estimate speed
        self.ball_history.append((ball_x, ball_y))
        if len(self.ball_history) > 6:
            self.ball_history.pop(0)

        shot_detected = False
        if self.shot_cooldown == 0 and len(self.ball_history) >= 4:
            speeds = []
            for i in range(1, len(self.ball_history)):
                x1, y1 = self.ball_history[i - 1]
                x2, y2 = self.ball_history[i]
                dx = x2 - x1
                dy = y2 - y1
                speeds.append((dx * dx + dy * dy) ** 0.5)

            max_speed = max(speeds)
            speed_jump = max_speed - min(speeds)
            speed_threshold = max(6.0, 0.005 * max(w, h))
            jump_threshold = speed_threshold * 0.4

            near_player = True
            if players:
                ball_cx, ball_cy = ball_x, ball_y
                closest = None
                for player in players:
                    px1, py1, px2, py2 = player['box']
                    pcx = (px1 + px2) / 2
                    pcy = (py1 + py2) / 2
                    dist = ((ball_cx - pcx) ** 2 + (ball_cy - pcy) ** 2) ** 0.5
                    if closest is None or dist < closest:
                        closest = dist

                max_dist = 0.25 * max(w, h)
                near_player = closest is not None and closest < max_dist

            if max_speed > speed_threshold and speed_jump > jump_threshold and near_player:
                shot_detected = True
                self.shot_cooldown = 15

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
