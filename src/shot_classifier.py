import numpy as np
import pickle
from src.utils import get_box_center, get_distance


class ShotClassifier:

    def __init__(self):
        self.slot_positions = {}
        self.shot_cooldown = {}
        self.speed_history = {}
        self.flow_history = {}
        self.prev_keypoints = {}

        try:
            with open('models/shot_classifier.pkl', 'rb') as f:
                self.clf = pickle.load(f)
            with open('models/shot_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("Shot classifier loaded!")
        except:
            self.clf = None
            self.scaler = None
            print("No classifier — using rule based")

    def get_nearest_slot(self, player_box, max_slots=4):
        cx, cy = get_box_center(player_box)

        min_dist = float('inf')
        best_slot = None

        for slot, (sx, sy) in self.slot_positions.items():
            dist = get_distance((cx, cy), (sx, sy))
            if dist < min_dist:
                min_dist = dist
                best_slot = slot

        if best_slot is not None and min_dist < 200:
            self.slot_positions[best_slot] = (cx, cy)
            return best_slot

        new_slot = len(self.slot_positions)
        if new_slot < max_slots:
            self.slot_positions[new_slot] = (cx, cy)
            return new_slot

        self.slot_positions[best_slot] = (cx, cy)
        return best_slot

    def is_shot_moment(self, slot_id, keypoints,
                       ball, player_box,
                       flow_magnitude=0.0):
        """
        Detect shot using:
        1. Wrist speed history
        2. Optical flow magnitude
        3. Ball proximity
        """
        if keypoints is None:
            return False

        if slot_id not in self.shot_cooldown:
            self.shot_cooldown[slot_id] = 0
        if slot_id not in self.speed_history:
            self.speed_history[slot_id] = []
        if slot_id not in self.flow_history:
            self.flow_history[slot_id] = []

        if self.shot_cooldown[slot_id] > 0:
            self.shot_cooldown[slot_id] -= 1
            return False

        # Ball proximity
        ball_near = False
        if ball:
            ball_cx, ball_cy = get_box_center(ball['box'])
            player_cx, player_cy = get_box_center(player_box)
            ball_dist = get_distance(
                (ball_cx, ball_cy),
                (player_cx, player_cy)
            )
            ball_near = ball_dist < 200

        # Wrist speed
        wrist_speed = 0
        if slot_id in self.prev_keypoints and \
           self.prev_keypoints[slot_id] is not None:
            if len(keypoints) > 21 and \
               len(self.prev_keypoints[slot_id]) > 21:
                # Right wrist = index 10 → pos 20,21
                curr_wx = keypoints[10*2]
                curr_wy = keypoints[10*2+1]
                prev_wx = self.prev_keypoints[slot_id][10*2]
                prev_wy = self.prev_keypoints[slot_id][10*2+1]
                wrist_speed = get_distance(
                    (curr_wx, curr_wy),
                    (prev_wx, prev_wy)
                )

        self.prev_keypoints[slot_id] = keypoints

        # Update histories
        self.speed_history[slot_id].append(wrist_speed)
        if len(self.speed_history[slot_id]) > 5:
            self.speed_history[slot_id].pop(0)

        self.flow_history[slot_id].append(flow_magnitude)
        if len(self.flow_history[slot_id]) > 5:
            self.flow_history[slot_id].pop(0)

        avg_speed = sum(self.speed_history[slot_id]) / \
                    len(self.speed_history[slot_id])

        avg_flow = sum(self.flow_history[slot_id]) / \
                   len(self.flow_history[slot_id])

        # Shot detection logic:
        # Strong signal: both flow and wrist speed high
        if avg_flow > 1.5 and avg_speed > 0.02:
            self.shot_cooldown[slot_id] = 15
            return True

        # Medium signal: ball near and some movement
        if ball_near and avg_speed > 0.02:
            self.shot_cooldown[slot_id] = 15
            return True

        # Strong wrist movement alone
        if avg_speed > 0.06:
            self.shot_cooldown[slot_id] = 15
            return True

        return False

    def extract_features(self, keypoints):
        if keypoints is None:
            return None

        if len(keypoints) < 26:
            return None

        def get_xy(idx):
            return keypoints[idx*2], keypoints[idx*2+1]

        l_sx, l_sy = get_xy(5)   # left shoulder
        r_sx, r_sy = get_xy(6)   # right shoulder
        l_ex, l_ey = get_xy(7)   # left elbow
        r_ex, r_ey = get_xy(8)   # right elbow
        l_wx, l_wy = get_xy(9)   # left wrist
        r_wx, r_wy = get_xy(10)  # right wrist
        l_hx, l_hy = get_xy(11)  # left hip
        r_hx, r_hy = get_xy(12)  # right hip

        body_cx = (l_sx + r_sx) / 2
        body_cy = (l_sy + r_sy) / 2
        sw = abs(r_sx - l_sx) + 1e-6

        hip_cx = (l_hx + r_hx) / 2
        hip_cy = (l_hy + r_hy) / 2

        r_wrist_rel_x = (r_wx - body_cx) / sw
        r_wrist_rel_y = (r_wy - body_cy) / sw
        l_wrist_rel_x = (l_wx - body_cx) / sw
        l_wrist_rel_y = (l_wy - body_cy) / sw

        r_elbow_rel_x = (r_ex - body_cx) / sw
        r_elbow_rel_y = (r_ey - body_cy) / sw
        l_elbow_rel_x = (l_ex - body_cx) / sw
        l_elbow_rel_y = (l_ey - body_cy) / sw

        r_wrist_above = (r_sy - r_wy) / sw
        l_wrist_above = (l_sy - l_wy) / sw

        r_arm_ext = np.sqrt(
            (r_wx - r_sx)**2 + (r_wy - r_sy)**2
        ) / sw

        l_arm_ext = np.sqrt(
            (l_wx - l_sx)**2 + (l_wy - l_sy)**2
        ) / sw

        shoulder_angle = np.arctan2(
            r_sy - l_sy, r_sx - l_sx
        )

        torso_height = abs(body_cy - hip_cy) / sw

        r_wrist_hip_x = (r_wx - hip_cx) / sw
        r_wrist_hip_y = (r_wy - hip_cy) / sw

        r_elbow_angle = np.arctan2(
            r_wy - r_ey, r_wx - r_ex
        )
        l_elbow_angle = np.arctan2(
            l_wy - l_ey, l_wx - l_ex
        )

        return [
            r_wrist_rel_x, r_wrist_rel_y,
            l_wrist_rel_x, l_wrist_rel_y,
            r_elbow_rel_x, r_elbow_rel_y,
            l_elbow_rel_x, l_elbow_rel_y,
            r_wrist_above, l_wrist_above,
            r_arm_ext, l_arm_ext,
            shoulder_angle, torso_height,
            r_wrist_hip_x, r_wrist_hip_y,
            r_elbow_angle, l_elbow_angle
        ]

    def classify(self, keypoints):
        if keypoints is None:
            return "unknown"

        if self.clf is not None and self.scaler is not None:
            try:
                features = self.extract_features(keypoints)
                if features is None:
                    return "unknown"
                fs = self.scaler.transform([features])
                return self.clf.predict(fs)[0]
            except Exception as e:
                pass

        # Rule based fallback
        if len(keypoints) < 22:
            return "unknown"

        def get_xy(idx):
            return keypoints[idx*2], keypoints[idx*2+1]

        r_wx, r_wy = get_xy(10)
        r_sx, r_sy = get_xy(6)
        l_sx, l_sy = get_xy(5)

        body_cx = (r_sx + l_sx) / 2
        sw = abs(r_sx - l_sx) + 1e-6
        wrist_above = (r_sy - r_wy) / sw

        if wrist_above > 0.5:
            return "smash"
        elif r_wx > body_cx:
            return "forehand"
        else:
            return "backhand"